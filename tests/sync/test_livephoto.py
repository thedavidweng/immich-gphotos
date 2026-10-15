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
