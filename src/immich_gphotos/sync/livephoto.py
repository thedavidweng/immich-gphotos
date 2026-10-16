"""Mux an Apple Live Photo (still + `.MOV`) into a Google Motion Photo.

In Immich an Apple Live Photo arrives as two paired assets: the primary
still (`.HEIC`/`.JPG`, `visibility: "timeline"`, carrying
`livePhotoVideoId`) and the companion motion (`.MOV`, `visibility:
"hidden"`). Uploading only the still loses the motion; uploading the `.MOV`
separately litters the timeline with 1.5-3s clips Google never pairs back.

This muxes the pair into a single file following the [Motion Photo format
1.0](https://developer.android.com/media/platform/motion-photo-format):
a primary still with Camera + Container XMP describing one `Primary`
image item and one `MotionPhoto` video item, the video bytes appended
after the still. Google Photos recognises the structure and renders it as
an interactive Motion Photo with sound.

Two still containers are supported, matching the two iPhone originals in
the wild:

- JPEG: the XMP goes in a new APP1 segment after SOI, the video is
  appended after EOI.
- HEIC/AVIF: the XMP goes in as a new `mime` item in the `meta` box (with
  an `iloc` extent pointing at it), and the video is wrapped in a trailing
  `mpvd` box whose 8-byte header is declared as the primary item's
  `Padding`, exactly per the spec's ISOBMFF section.

Anything else -- a still that is neither JPEG nor a parsable HEIC/AVIF, a
`meta` box without `iinf`/`iloc`, an entry version this writer does not
emit, an `iloc` entry not addressable as a plain file offset -- falls back
to still-only upload, as before, rather than producing a corrupt file. The
old `GCamera:MicroVideo` attributes are deliberately not written: deleted
in 1.0, readers must ignore them.
"""

import struct

# Set when the companion `.MOV` was folded into its still's upload, so the
# hidden clip is accounted for instead of sitting in `ineligible: hidden`
# forever alongside thousands of identical rows.
LIVE_PHOTO_PAIRED_REASON = "live_photo_paired"

_JPEG_SOI = b"\xff\xd8"
_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"

_CAMERA_NS = "http://ns.google.com/photos/1.0/camera/"
_CONTAINER_NS = "http://ns.google.com/photos/1.0/container/"
_ITEM_NS = "http://ns.google.com/photos/1.0/container/item/"

# Cap the motion clip we will fold in. Apple Live clips are ~1.5-3s
# (a few MB); anything far larger is not a Live clip and muxing it would
# bloat scratch and the upload for no motion-photo benefit.
MAX_VIDEO_BYTES = 64 * 1024 * 1024


def is_jpeg(data: bytes) -> bool:
    """True when `data` starts with the JPEG SOI marker."""
    return data[:2] == _JPEG_SOI


def looks_like_video(data: bytes) -> bool:
    """True for bytes plausibly an MP4/MOV container.

    Checks the first 32 bytes for an `ftyp`, `moov`, `mdat`, `free` or
    `wide` box -- enough to refuse muxing garbage while accepting every
    real `.MOV`/`.MP4` companion. Anything else falls back to still-only
    rather than producing a file Google would reject.
    """
    head = data[:32]
    return any(tag in head for tag in (b"ftyp", b"moov", b"mdat", b"free", b"wide"))


def _xmp_packet(
    video_length: int, video_mime: str, image_mime: str = "image/jpeg", padding: int | None = None
) -> bytes:
    """The Camera + Container XMP describing one still + one motion item.

    `video_length` is the exact byte length of the appended video (the
    secondary item's `Length`); the primary item's length is 0 by convention
    (readers parse the still to find its end). `padding` is the ISOBMFF-only
    primary-item `Padding`: 8 for HEIC/AVIF (the `mpvd` header sitting
    between the image and the video), omitted for JPEG. Presentation
    timestamp is -1 (unset), so readers use the middle of the video track.
    """
    padding_attr = f' Item:Padding="{padding}"' if padding is not None else ""
    xml = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:Camera="{_CAMERA_NS}" '
        'Camera:MotionPhoto="1" Camera:MotionPhotoVersion="1" '
        'Camera:MotionPhotoPresentationTimestampUs="-1"/>'
        f'<rdf:Description rdf:about="" xmlns:Container="{_CONTAINER_NS}">'
        "<Container:Directory>"
        '<rdf:Seq xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:li rdf:parseType="Resource" xmlns:Item="{_ITEM_NS}" '
        f'Item:Mime="{image_mime}" Item:Semantic="Primary" Item:Length="0"{padding_attr}/>'
        f'<rdf:li rdf:parseType="Resource" xmlns:Item="{_ITEM_NS}" '
        f'Item:Mime="{video_mime}" Item:Semantic="MotionPhoto" '
        f'Item:Length="{video_length}"/>'
        "</rdf:Seq>"
        "</Container:Directory>"
        "</rdf:Description>"
        "</rdf:RDF>"
        "</x:xmpmeta>"
    )
    return xml.encode("utf-8")


def _app1_segment(packet: bytes) -> bytes:
    """One JPEG APP1 segment carrying `packet` (header + XMP).

    Segment layout: `FF E1 LEN(2 bytes BE, including the 2 length bytes)`
    + payload. LEN must fit in 16 bits; our packet is ~1KB, far below the
    ~64KB segment limit, so no extended-XMP chunking is needed.
    """
    payload = _XMP_HEADER + packet
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def build_motion_photo_jpeg(
    still_jpeg: bytes, video_bytes: bytes, video_mime: str = "video/quicktime"
) -> bytes:
    """Return `still_jpeg` with Container/Camera XMP inserted and `video_bytes` appended.

    The XMP APP1 is inserted immediately after the SOI marker (APP segments
    must precede the scan data), leaving the original image bytes -- EXIF
    included -- untouched; any viewer ignoring XMP still sees the exact
    original still. Raises `ValueError` for anything that should fall back
    to still-only instead of producing a suspect file.
    """
    if not is_jpeg(still_jpeg):
        raise ValueError("still is not a JPEG")
    if not video_bytes:
        raise ValueError("companion video is empty")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise ValueError(f"companion video too large: {len(video_bytes)} bytes")
    if not looks_like_video(video_bytes):
        raise ValueError("companion bytes do not look like MP4/MOV")
    if b"\xff\xd9" not in still_jpeg:
        raise ValueError("still JPEG has no EOI marker")
    packet = _xmp_packet(len(video_bytes), video_mime)
    segment = _app1_segment(packet)
    still_with_xmp = still_jpeg[:2] + segment + still_jpeg[2:]
    return still_with_xmp + video_bytes


def video_mime_for_filename(filename: str) -> str:
    """Container MIME for the secondary item: QuickTime for `.MOV` (Apple
    Live clips), MP4 otherwise."""
    return "video/quicktime" if filename.lower().endswith(".mov") else "video/mp4"


# ---------------------------------------------------------------------------
# ISOBMFF (HEIC/AVIF) Motion Photos
#
# The still keeps its boxes byte-identical except `meta`, which grows by one
# XMP `mime` item; the XMP bytes live in a new trailing `mdat`, and the video
# in a trailing `mpvd` box after that. Every pre-existing file-offset
# reference past the old end of `meta` is shifted by the growth, in the same
# pass that appends the new entries. Anything this writer cannot model
# exactly raises ValueError so the caller falls back to still-only.
# ---------------------------------------------------------------------------

_HEIC_BRANDS = frozenset({"heic", "heix", "hevc", "hevx", "heim", "heis", "hevm", "hevs"})
_AVIF_BRANDS = frozenset({"avif", "avis"})
_GENERIC_HEIF_BRANDS = frozenset({"mif1", "msf1"})

_XMP_CONTENT_TYPE = b"application/rdf+xml"


def _u32(data: bytes, pos: int) -> int:
    return struct.unpack_from(">I", data, pos)[0]


def _u16(data: bytes, pos: int) -> int:
    return struct.unpack_from(">H", data, pos)[0]


def _read_box(data: bytes, pos: int) -> tuple[bytes, int, int, int]:
    """Parse one box header at `pos`: (type, box_start, box_size, header_len).

    Handles the 32-bit size, the `largesize` form (`size == 1`), and the
    to-EOF form (`size == 0`, top-level only). Raises ValueError on anything
    malformed or truncated -- every caller treats that as "not muxable".
    """
    if pos + 8 > len(data):
        raise ValueError("truncated box header")
    size = _u32(data, pos)
    box_type = data[pos + 4 : pos + 8]
    if size == 1:
        if pos + 16 > len(data):
            raise ValueError("truncated largesize")
        size = struct.unpack_from(">Q", data, pos + 8)[0]
        header_len = 16
    elif size == 0:
        size = len(data) - pos
        header_len = 8
    else:
        header_len = 8
        if size < 8:
            raise ValueError(f"impossible box size {size} for {box_type!r}")
    if size > len(data) - pos:
        raise ValueError(f"{box_type!r} box overruns its buffer")
    return box_type, pos, size, header_len


def _top_level_boxes(data: bytes) -> list[tuple[bytes, int, int]]:
    """Split ISOBMFF bytes into (type, start, size) top-level boxes."""
    boxes: list[tuple[bytes, int, int]] = []
    pos = 0
    if not data:
        raise ValueError("empty file")
    while pos < len(data):
        box_type, start, size, _ = _read_box(data, pos)
        boxes.append((box_type, start, size))
        pos = start + size
    return boxes


def isobmff_image_kind(data: bytes) -> str | None:
    """`"heic"`, `"avif"`, or None for bytes that are not a still we can mux.

    Reads only the `ftyp` brands: a HEIC-family (or generic-HEIF-with-HEIC
    compatible) major/compatible brand means HEIC, an AVIF brand means AVIF.
    Anything unparseable -- or a video `ftyp` like `mp42`/`qt` -- is None.
    """
    try:
        box_type, _, size, header_len = _read_box(data, 0)
    except ValueError:
        return None
    if box_type != b"ftyp" or size < header_len + 8:
        return None
    payload = data[header_len:size]
    major = payload[0:4].decode("latin1")
    compat = {payload[i : i + 4].decode("latin1") for i in range(4, len(payload) - 3, 4)}
    if major in _HEIC_BRANDS or (major in _GENERIC_HEIF_BRANDS and compat & _HEIC_BRANDS):
        return "heic"
    if major in _AVIF_BRANDS:
        return "avif"
    return None


def image_mime_for_kind(kind: str) -> str:
    """Primary-item MIME for an ISOBMFF still kind."""
    return "image/avif" if kind == "avif" else "image/heic"


def _read_cstr(data: bytes, pos: int, end: int) -> tuple[bytes, int]:
    """Read one NUL-terminated string; returns (value, next_pos)."""
    stop = data.find(b"\x00", pos, end)
    if stop < 0:
        raise ValueError("unterminated string in item info")
    return data[pos:stop], stop + 1


def _parse_iinf(payload: bytes) -> tuple[int, list[tuple[int, bytes]], int]:
    """Parse an `iinf` payload: (version, [(item_id, raw_entry)], max_id).

    Entries are returned raw so the rebuild can copy them verbatim; only the
    ids are interpreted. `content_encoding`-carrying entries (v2+) make entry
    boundaries ambiguous, so a payload that does not parse exactly to its end
    without one is rejected -- those files fall back to still-only.
    iinf version 0 (16-bit ids, free-form string fields) predates reliable
    XMP carriage and is rejected for the same reason.
    """
    if len(payload) < 6:
        raise ValueError("truncated iinf")
    version = payload[0]
    if version == 0:
        raise ValueError("iinf version 0 is not supported for XMP insertion")
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported iinf version {version}")
    (count,) = struct.unpack_from(">I", payload, 4)
    entries: list[tuple[int, bytes]] = []
    max_id = 0
    pos = 8
    for _ in range(count):
        start = pos
        item_id = _u16(payload, pos) if version < 3 else _u32(payload, pos)
        pos += 2 if version < 3 else 4
        pos += 2  # item_protection_index
        if pos + 4 > len(payload):
            raise ValueError("truncated infe entry")
        pos += 4  # item_type
        _, pos = _read_cstr(payload, pos, len(payload))  # item_name
        if version >= 2:
            _, pos = _read_cstr(payload, pos, len(payload))  # content_type
        entries.append((item_id, payload[start:pos]))
        max_id = max(max_id, item_id)
    if pos != len(payload) or len(entries) != count:
        raise ValueError("iinf entries do not fill their box exactly")
    return version, entries, max_id


def _build_infe(version: int, item_id: int) -> bytes:
    """One XMP `mime` item entry in the container's own iinf version."""
    head = struct.pack(">I", item_id) if version >= 3 else struct.pack(">H", item_id)
    head += struct.pack(">H", 0) + b"mime" + b"\x00"
    if version >= 2:
        head += _XMP_CONTENT_TYPE + b"\x00"
    return head


def _parse_iloc(payload: bytes) -> dict:
    """Parse an `iloc` payload into a re-encodable dict.

    Every entry becomes (item_id, method|None, ref, base, [(offset, length)]).
    v0 has no construction_method (treated as file offsets, method 0).
    Raises ValueError for versions other than 0-2 or any truncation.
    """
    if len(payload) < 8:
        raise ValueError("truncated iloc")
    version = payload[0]
    if version not in (0, 1, 2):
        raise ValueError(f"unsupported iloc version {version}")
    offset_size = payload[4] >> 4
    length_size = payload[4] & 15
    base_offset_size = payload[5] >> 4
    index_size = payload[5] & 15
    for name, size in (("offset", offset_size), ("length", length_size), ("base_offset", base_offset_size)):
        if size not in (0, 4, 8):
            raise ValueError(f"unsupported iloc {name}_size {size}")
    if version in (1, 2) and index_size not in (0, 4, 8):
        raise ValueError(f"unsupported iloc index_size {index_size}")
    pos = 6
    count = _u16(payload, pos) if version < 2 else _u32(payload, pos)
    pos += 2 if version < 2 else 4
    entries = []
    for _ in range(count):
        item_id = _u16(payload, pos) if version < 2 else _u32(payload, pos)
        pos += 2 if version < 2 else 4
        method: int | None = None
        if version in (1, 2):
            method = _u16(payload, pos)
            pos += 2
        ref = _u16(payload, pos)
        pos += 2
        base = int.from_bytes(payload[pos : pos + base_offset_size], "big") if base_offset_size else 0
        pos += base_offset_size
        extent_count = _u16(payload, pos)
        pos += 2
        extents = []
        for _ in range(extent_count):
            if version in (1, 2) and index_size:
                pos += index_size
            off = int.from_bytes(payload[pos : pos + offset_size], "big") if offset_size else 0
            pos += offset_size
            ln = int.from_bytes(payload[pos : pos + length_size], "big") if length_size else 0
            pos += length_size
            extents.append((off, ln))
        entries.append((item_id, method, ref, base, extents))
    if pos != len(payload):
        raise ValueError("iloc entries do not fill their box exactly")
    return {
        "version": version,
        "flags": (payload[1] << 16) | (payload[2] << 8) | payload[3],
        "offset_size": offset_size,
        "length_size": length_size,
        "base_offset_size": base_offset_size,
        "index_size": index_size,
        "entries": entries,
    }


def _encode_iloc_entry(parsed: dict, entry: tuple) -> bytes:
    """Encode one iloc entry with the container's own version and sizes."""
    version = parsed["version"]
    item_id, method, ref, base, extents = entry
    out = struct.pack(">I" if version >= 2 else ">H", item_id)
    if version in (1, 2):
        out += struct.pack(">H", method if method is not None else 0)
    out += struct.pack(">H", ref)
    if parsed["base_offset_size"]:
        out += base.to_bytes(parsed["base_offset_size"], "big")
    out += struct.pack(">H", len(extents))
    for off, ln in extents:
        if version in (1, 2) and parsed["index_size"]:
            out += (0).to_bytes(parsed["index_size"], "big")
        if parsed["offset_size"]:
            out += off.to_bytes(parsed["offset_size"], "big")
        if parsed["length_size"]:
            out += ln.to_bytes(parsed["length_size"], "big")
    return out


def _encode_iloc(parsed: dict) -> bytes:
    """Re-encode a parsed iloc payload (header fields + entries)."""
    version = parsed["version"]
    out = struct.pack(">I", (version << 24) | parsed["flags"])
    out += bytes([(parsed["offset_size"] << 4) | parsed["length_size"]])
    out += bytes([(parsed["base_offset_size"] << 4) | parsed["index_size"]])
    out += struct.pack(">I" if version >= 2 else ">H", len(parsed["entries"]))
    for entry in parsed["entries"]:
        out += _encode_iloc_entry(parsed, entry)
    return out


def _rebuild_meta_with_xmp(
    meta_box: bytes, xmp: bytes, xmp_offset: int | None, meta_start: int
) -> tuple[bytes, int]:
    """Return (new_meta_box, new_item_id) with an XMP `mime` item added.

    `xmp_offset` is the absolute file offset the XMP payload will live at;
    `None` is a measuring pass (the entry is staged with a dummy offset --
    entry encodings are value-invariant in length, so the growth is exact --
    and the placement check is skipped). `meta_start` locates the box so
    pre-existing file offsets past the old end of `meta` can be shifted by
    the growth. Only construction_method-0, same-file references are
    shifted; anything else present means the file is left alone
    (ValueError).
    """
    version_flags = meta_box[8:12]
    children: list[tuple[bytes, bytes]] = []  # (type, full_box_bytes)
    pos = 12
    while pos < len(meta_box):
        box_type, _, size, _ = _read_box(meta_box, pos)
        children.append((box_type, meta_box[pos : pos + size]))
        pos += size
    iinf_raw = iloc_raw = None
    for box_type, raw in children:
        if box_type == b"iinf" and iinf_raw is None:
            iinf_raw = raw
        elif box_type == b"iloc" and iloc_raw is None:
            iloc_raw = raw
    if iinf_raw is None or iloc_raw is None:
        raise ValueError("meta has no iinf/iloc pair")
    iinf_version, entries, max_id = _parse_iinf(iinf_raw[8:])
    new_id = max_id + 1
    if new_id > (0xFFFFFFFF if iinf_version >= 3 else 0xFFFF):
        raise ValueError("item id space exhausted")
    new_infe = _build_infe(iinf_version, new_id)
    new_iinf_payload = iinf_raw[8:12] + struct.pack(">I", len(entries) + 1)
    for _, raw_entry in entries:
        new_iinf_payload += raw_entry
    new_iinf_payload += new_infe
    new_iinf = struct.pack(">I", 8 + len(new_iinf_payload)) + b"iinf" + new_iinf_payload

    iloc = _parse_iloc(iloc_raw[8:])
    if iloc["offset_size"] == 0 or iloc["length_size"] == 0:
        raise ValueError("iloc cannot address the new extent")
    meta_end_old = meta_start + len(meta_box)
    # The rebuilt meta's own growth is computable before any offset is known:
    # entry encodings are value-invariant in length, so measure the new entry
    # with a dummy offset, derive the true delta, then encode once for real.
    new_entry_stub = (new_id, 0 if iloc["version"] in (1, 2) else None, 0, 0, [(0, len(xmp))])
    try:
        iloc_growth = len(_encode_iloc_entry(iloc, new_entry_stub))
    except OverflowError as exc:
        raise ValueError(f"new iloc entry does not fit container sizes: {exc}") from exc
    delta = (len(new_iinf) - len(iinf_raw)) + iloc_growth
    # Shift every same-file reference at or past the old end of meta. Other
    # construction methods (idat-relative) or external references are left
    # alone -- but their very presence means meta growth could invalidate
    # them, so they refuse the mux instead.
    shifted = []
    for item_id, method, ref, base, extents in iloc["entries"]:
        if method is not None and method != 0:
            raise ValueError(f"unsupported iloc construction_method {method}")
        if ref == 0:
            if base >= meta_end_old and base != 0:
                base += delta
            extents = [(off + delta if off >= meta_end_old else off, ln) for off, ln in extents]
        shifted.append((item_id, method, ref, base, extents))
    if xmp_offset is not None and xmp_offset < meta_end_old + delta:
        raise ValueError("XMP offset precedes the rebuilt meta")
    real_offset = xmp_offset if xmp_offset is not None else 0
    try:
        new_entry = (new_id, 0 if iloc["version"] in (1, 2) else None, 0, 0, [(real_offset, len(xmp))])
        _encode_iloc_entry(iloc, new_entry)
    except OverflowError as exc:
        raise ValueError(f"XMP extent does not fit container sizes: {exc}") from exc
    iloc["entries"] = [*shifted, new_entry]
    new_iloc_payload = _encode_iloc(iloc)
    new_iloc = struct.pack(">I", 8 + len(new_iloc_payload)) + b"iloc" + new_iloc_payload
    # Rebuild meta with the two replaced children, preserving order.
    out = bytearray(version_flags)
    replaced_iinf = replaced_iloc = False
    for box_type, raw in children:
        if box_type == b"iinf" and not replaced_iinf:
            out += new_iinf
            replaced_iinf = True
        elif box_type == b"iloc" and not replaced_iloc:
            out += new_iloc
            replaced_iloc = True
        else:
            out += raw
    new_meta = struct.pack(">I", 8 + len(out)) + b"meta" + bytes(out)
    if len(new_meta) - len(meta_box) != delta:
        # Internal consistency: size growth must equal the measured delta.
        raise ValueError("meta rebuild size mismatch")
    return new_meta, new_id


def _mpvd_box(video_bytes: bytes) -> bytes:
    """The trailing Motion Photo Video Data box wrapping the video bytes."""
    size = len(video_bytes) + 8
    if size < 2**32:
        return struct.pack(">I", size) + b"mpvd" + video_bytes
    return struct.pack(">I", 1) + b"mpvd" + struct.pack(">Q", size + 8) + video_bytes


def build_motion_photo_isobmff(
    still: bytes, video_bytes: bytes, image_mime: str = "image/heic", video_mime: str = "video/quicktime"
) -> bytes:
    """Return an HEIC/AVIF still muxed with `video_bytes` as a Motion Photo.

    Layout: original boxes with a rebuilt `meta` (one XMP `mime` item added),
    then an `mdat` holding the XMP packet, then an `mpvd` box holding the
    video. Raises ValueError for anything that should fall back to still-only.
    """
    if not video_bytes:
        raise ValueError("companion video is empty")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise ValueError(f"companion video too large: {len(video_bytes)} bytes")
    if not looks_like_video(video_bytes):
        raise ValueError("companion bytes do not look like MP4/MOV")
    boxes = _top_level_boxes(still)
    meta_index = next((i for i, (t, _, _) in enumerate(boxes) if t == b"meta"), None)
    if meta_index is None:
        raise ValueError("still has no top-level meta box")
    packet = _xmp_packet(len(video_bytes), video_mime, image_mime, padding=8)
    # Prefix length (everything up to and including the rebuilt meta) fixes
    # the absolute offset of the appended XMP payload. Meta growth is
    # value-invariant, so one measuring pass plus one final pass converges.
    _, meta_start, meta_size = boxes[meta_index]
    meta_raw = still[meta_start : meta_start + meta_size]
    measuring_meta, _ = _rebuild_meta_with_xmp(meta_raw, packet, None, meta_start)
    prefix_len = len(still) - meta_size + len(measuring_meta)
    xmp_offset = prefix_len + 8  # past the XMP mdat header
    new_meta, _ = _rebuild_meta_with_xmp(meta_raw, packet, xmp_offset, meta_start)
    if len(new_meta) != len(measuring_meta):
        raise ValueError("meta rebuild is not size-stable")
    out = bytearray()
    for i, (_, start, size) in enumerate(boxes):
        out += new_meta if i == meta_index else still[start : start + size]
    if len(out) != xmp_offset - 8:
        raise ValueError("rebuilt prefix length mismatch")
    out += struct.pack(">I", len(packet) + 8) + b"mdat" + packet
    out += _mpvd_box(video_bytes)
    return bytes(out)
