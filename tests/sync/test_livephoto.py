import struct

import pytest

from immich_gphotos.sync import livephoto

# Minimal valid JPEG: SOI + APP0 (JFIF) + EOI. Real stills are larger; the
# muxer only needs SOI ... EOI with an insertion point after SOI.
STILL = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 32 + b"\xff\xd9"
# Minimal MP4/MOV-looking bytes: size + 'ftyp' + payload.
VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _app1_packets(muxed: bytes) -> list[bytes]:
    """Every APP1 payload between SOI and the original still's third byte."""
    assert muxed[:2] == b"\xff\xd8"
    packets = []
    pos = 2
    while muxed[pos : pos + 2] == b"\xff\xe1":
        length = struct.unpack(">H", muxed[pos + 2 : pos + 4])[0]
        packets.append(muxed[pos + 4 : pos + 2 + length])
        pos += 2 + length
    return packets


def test_muxed_file_is_still_then_video_with_xmp_between():
    muxed = livephoto.build_motion_photo_jpeg(STILL, VIDEO)

    # SOI, then our APP1, then the original still bytes verbatim, then video.
    assert muxed[:2] == b"\xff\xd8"
    packets = _app1_packets(muxed)
    assert len(packets) == 1
    assert packets[0].startswith(b"http://ns.adobe.com/xap/1.0/\x00")
    xmp = packets[0].split(b"\x00", 1)[1].decode()
    assert 'Camera:MotionPhoto="1"' in xmp
    assert 'Camera:MotionPhotoVersion="1"' in xmp
    assert f'Item:Length="{len(VIDEO)}"' in xmp
    assert 'Item:Semantic="MotionPhoto"' in xmp
    assert 'Item:Semantic="Primary"' in xmp
    # Original still intact after our segment, video appended at the end.
    assert STILL[2:] in muxed
    assert muxed.endswith(VIDEO)
    assert muxed.index(STILL[2:]) < muxed.index(VIDEO)


def test_muxed_file_keeps_an_existing_exif_segment():
    packets = _app1_packets(livephoto.build_motion_photo_jpeg(STILL, VIDEO))
    assert len(packets) == 1  # ours, inserted first; the JFIF APP0 follows untouched


def test_still_must_be_a_jpeg():
    with pytest.raises(ValueError):
        livephoto.build_motion_photo_jpeg(b"not a jpeg", VIDEO)


def test_video_must_not_be_empty_or_huge_or_garbage():
    with pytest.raises(ValueError):
        livephoto.build_motion_photo_jpeg(STILL, b"")
    with pytest.raises(ValueError):
        livephoto.build_motion_photo_jpeg(STILL, b"x" * (livephoto.MAX_VIDEO_BYTES + 1))
    with pytest.raises(ValueError):
        livephoto.build_motion_photo_jpeg(STILL, b"garbage" * 16)


def test_video_mime_follows_the_companion_extension():
    assert livephoto.video_mime_for_filename("IMG_1.MOV") == "video/quicktime"
    assert livephoto.video_mime_for_filename("clip.mp4") == "video/mp4"
    xmp = livephoto._xmp_packet(10, "video/quicktime").decode()
    assert 'Item:Mime="video/quicktime"' in xmp


# ---------------------------------------------------------------------------
# ISOBMFF (HEIC/AVIF) stills
# ---------------------------------------------------------------------------


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def make_heic_still() -> tuple[bytes, bytes, bytes]:
    """A minimal but structurally valid HEIC still.

    Layout: ftyp + meta(hdlr, pitm, iinf[v2: hvc1 + thmb], iloc[v1]) + mdat.
    Returns (file_bytes, image_payload, thumb_payload) so tests can assert
    the payloads survive the mux byte-identical at shifted offsets.
    """
    image = bytes(range(64))
    thumb = bytes(range(16))
    ftyp = _box(b"ftyp", b"heic" + struct.pack(">I", 0) + b"heic" + b"mif1")
    hdlr = _box(b"hdlr", b"\x00\x00\x00\x00" + b"\x00\x00\x00\x00" + b"pict" + b"\x00" * 12 + b"\x00")
    pitm = _box(b"pitm", b"\x00\x00\x00\x00" + struct.pack(">H", 1))
    infe1 = struct.pack(">H", 1) + struct.pack(">H", 0) + b"hvc1" + b"Image\x00" + b"\x00"
    infe2 = struct.pack(">H", 2) + struct.pack(">H", 0) + b"thmb" + b"Thumb\x00" + b"\x00"
    iinf = _box(b"iinf", b"\x02\x00\x00\x00" + struct.pack(">I", 2) + infe1 + infe2)
    # iloc offsets are filled in once the layout is known: meta comes before
    # mdat, so the payloads sit at meta_end + 8 (+0 for image, +64 for thumb).
    meta_without_iloc = b"\x00\x00\x00\x00" + hdlr + pitm + iinf
    meta_size = 8 + len(meta_without_iloc) + 56  # iloc box is 56 bytes (see below)
    mdat_start = len(ftyp) + meta_size
    img_off = mdat_start + 8
    thm_off = img_off + len(image)
    iloc_payload = (
        b"\x01\x00\x00\x00"
        + bytes([(4 << 4) | 4])
        + bytes([(4 << 4) | 0])
        + struct.pack(">H", 2)
        + struct.pack(">H", 1)
        + struct.pack(">H", 0)
        + struct.pack(">H", 0)
        + struct.pack(">I", 0)
        + struct.pack(">H", 1)
        + struct.pack(">I", img_off)
        + struct.pack(">I", len(image))
        + struct.pack(">H", 2)
        + struct.pack(">H", 0)
        + struct.pack(">H", 0)
        + struct.pack(">I", 0)
        + struct.pack(">H", 1)
        + struct.pack(">I", thm_off)
        + struct.pack(">I", len(thumb))
    )
    assert len(iloc_payload) == 48
    iloc = _box(b"iloc", iloc_payload)
    meta = _box(b"meta", b"\x00\x00\x00\x00" + hdlr + pitm + iinf + iloc)
    assert len(meta) == meta_size
    mdat = _box(b"mdat", image + thumb)
    return ftyp + meta + mdat, image, thumb


def _parse_top_boxes(data: bytes) -> list[tuple[bytes, int, int]]:
    boxes = []
    pos = 0
    while pos < len(data):
        size = struct.unpack_from(">I", data, pos)[0]
        boxes.append((data[pos + 4 : pos + 8], pos, size))
        pos += size
    return boxes


def _meta_children(meta_box: bytes) -> dict[bytes, bytes]:
    children: dict[bytes, list[bytes]] = {}
    pos = 12
    while pos < len(meta_box):
        size = struct.unpack_from(">I", meta_box, pos)[0]
        children.setdefault(meta_box[pos + 4 : pos + 8], []).append(meta_box[pos : pos + size])
        pos += size
    return children


def test_heic_detection_by_brand():
    heic, _, _ = make_heic_still()
    assert livephoto.isobmff_image_kind(heic) == "heic"
    assert livephoto.isobmff_image_kind(STILL) is None
    assert livephoto.isobmff_image_kind(VIDEO) is None
    assert livephoto.isobmff_image_kind(b"junk") is None
    avif = heic.replace(b"heic", b"avif", 1)
    assert livephoto.isobmff_image_kind(avif) == "avif"


def _iloc_entries(iloc_box: bytes) -> list[tuple[int, int, int]]:
    """Parse a v1 iloc box into (item_id, extent_offset, extent_length)."""
    payload = iloc_box[8:]
    assert payload[0] == 1
    pos = 8
    out = []
    for _ in range(struct.unpack_from(">H", payload, 6)[0]):
        item_id = struct.unpack_from(">H", payload, pos)[0]
        pos += 2 + 2 + 2 + 4  # id, method, ref, base_offset(4)
        assert struct.unpack_from(">H", payload, pos)[0] == 1
        pos += 2
        off = struct.unpack_from(">I", payload, pos)[0]
        ln = struct.unpack_from(">I", payload, pos + 4)[0]
        pos += 8
        out.append((item_id, off, ln))
    assert pos == len(payload)
    return out


def _iinf_ids(iinf_box: bytes) -> list[int]:
    payload = iinf_box[8:]
    assert payload[0] == 2
    count = struct.unpack_from(">I", payload, 4)[0]
    ids = []
    pos = 8
    for _ in range(count):
        ids.append(struct.unpack_from(">H", payload, pos)[0])
        pos += 2 + 2 + 4  # id, protection, type
        for _ in range(2):  # name, content_type
            pos = payload.index(b"\x00", pos) + 1
    assert pos == len(payload)
    return ids


def test_heic_mux_round_trips_offsets_and_metadata():
    heic, image, thumb = make_heic_still()
    old_boxes = {t: (s, z) for t, s, z in _parse_top_boxes(heic)}
    muxed = livephoto.build_motion_photo_isobmff(heic, VIDEO, "image/heic", "video/quicktime")

    boxes = {t: (s, z) for t, s, z in _parse_top_boxes(muxed)}
    assert set(boxes) == {b"ftyp", b"meta", b"mdat", b"mpvd"}
    # Original ftyp untouched; video wrapped byte-identical in mpvd.
    assert muxed[: old_boxes[b"ftyp"][1]] == heic[: old_boxes[b"ftyp"][1]]
    mpvd_start, mpvd_size = boxes[b"mpvd"]
    assert muxed[mpvd_start + 8 : mpvd_start + mpvd_size] == VIDEO

    meta_start, meta_size = boxes[b"meta"]
    children = _meta_children(muxed[meta_start : meta_start + meta_size])
    # The XMP item was added on top of the two pre-existing items.
    assert _iinf_ids(children[b"iinf"][0]) == [1, 2, 3]
    entries = _iloc_entries(children[b"iloc"][0])
    assert [e[0] for e in entries] == [1, 2, 3]

    # The XMP mdat is the second mdat, immediately before mpvd; the new
    # extent addresses its payload exactly.
    mdats = [(s, z) for t, s, z in _parse_top_boxes(muxed) if t == b"mdat"]
    assert len(mdats) == 2
    assert mdats[1][0] + mdats[1][1] == mpvd_start
    xmp_payload = muxed[mdats[1][0] + 8 : mdats[1][0] + mdats[1][1]]
    assert entries[2][1] == mdats[1][0] + 8
    assert entries[2][2] == len(xmp_payload)

    xmp = xmp_payload.decode()
    assert 'Camera:MotionPhoto="1"' in xmp
    assert 'Item:Mime="image/heic"' in xmp
    assert 'Item:Mime="video/quicktime"' in xmp
    assert 'Item:Padding="8"' in xmp
    assert f'Item:Length="{len(VIDEO)}"' in xmp

    # Pre-existing payloads survived byte-identical, shifted past the grown
    # meta by exactly the growth.
    delta = meta_size - old_boxes[b"meta"][1]
    assert delta > 0
    assert muxed[entries[0][1] : entries[0][1] + entries[0][2]] == image
    assert muxed[entries[1][1] : entries[1][1] + entries[1][2]] == thumb
    old_img_off = old_boxes[b"meta"][0] + old_boxes[b"meta"][1] + 8
    assert entries[0][1] == old_img_off + delta


def test_heic_mux_rejects_unparsable_inputs():
    heic, _, _ = make_heic_still()
    with pytest.raises(ValueError):
        livephoto.build_motion_photo_isobmff(b"\x00\x00\x00\x10ftypmp42" + b"\x00" * 8, VIDEO)
    with pytest.raises(ValueError):
        livephoto.build_motion_photo_isobmff(heic, b"")
    with pytest.raises(ValueError):
        livephoto.build_motion_photo_isobmff(heic, b"garbage" * 16)
    with pytest.raises(ValueError):
        livephoto.build_motion_photo_isobmff(heic, b"x" * (livephoto.MAX_VIDEO_BYTES + 1))
