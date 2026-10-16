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

Only JPEG stills are muxed here. HEIC/AVIF Motion Photos are valid per the
same spec (via an `mpvd` box and `Padding: 8`), but packing XMP and the
video box into ISOBMFF without a dedicated writer is a separate project;
HEIC stills fall back to still-only upload, as before, rather than
producing a corrupt file. The old `GCamera:MicroVideo` attributes are
deliberately not written: deleted in 1.0, readers must ignore them.
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


def _xmp_packet(video_length: int, video_mime: str) -> bytes:
    """The Camera + Container XMP describing one still + one motion item.

    `video_length` is the exact byte length of the appended video (the
    secondary item's `Length`); the primary item's length is 0 by convention
    (readers parse the still to find its end). Presentation timestamp is -1
    (unset), so readers use the middle of the video track.
    """
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
        'Item:Mime="image/jpeg" Item:Semantic="Primary" Item:Length="0"/>'
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
