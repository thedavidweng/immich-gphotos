"""Opt-in smoke test against a REAL Google Photos account.

Skipped unless all of these are set (never in CI):

    IGP_GOOGLE_SMOKE=1
    IGP_GOOGLE_AUTH_DATA=<gpmc auth string of a throwaway account>
    IGP_GOOGLE_SMOKE_STILL=/path/to/IMG_xxxx.JPG   (a real Live Photo still)
    IGP_GOOGLE_SMOKE_VIDEO=/path/to/IMG_xxxx.MOV   (its paired clip)

It builds a derived JPEG Motion Photo from the pair in a temp dir (the inputs
are only read), uploads it, checks Google reports it by the derived SHA-1 and
NOT by the original still's SHA-1, then moves it to the Google trash. Whether
Google *plays* the motion still has to be checked by eye in the app; this
test cannot assert that.
"""

import os
from pathlib import Path

import pytest

from immich_gphotos.sync import livephoto

REQUIRED = ("IGP_GOOGLE_AUTH_DATA", "IGP_GOOGLE_SMOKE_STILL", "IGP_GOOGLE_SMOKE_VIDEO")

pytestmark = pytest.mark.skipif(
    os.environ.get("IGP_GOOGLE_SMOKE") != "1" or not all(os.environ.get(k) for k in REQUIRED),
    reason="real-Google smoke test; set IGP_GOOGLE_SMOKE=1 and " + ", ".join(REQUIRED),
)


def test_derived_motion_photo_round_trips_by_its_own_hash(tmp_path):
    from immich_gphotos.gphotos.client import GpmcClient

    still = Path(os.environ["IGP_GOOGLE_SMOKE_STILL"])
    video = Path(os.environ["IGP_GOOGLE_SMOKE_VIDEO"])
    before = (still.stat().st_mtime_ns, video.stat().st_mtime_ns)
    derived = tmp_path / ("MP_" + still.name)
    derived_sha1 = livephoto.write_motion_photo_jpeg(still, video, derived, "video/quicktime")
    still_sha1 = livephoto.sha1_b64_of_file(still)
    assert derived_sha1 != still_sha1

    client = GpmcClient(os.environ["IGP_GOOGLE_AUTH_DATA"])
    media_key = client.upload(derived, checksum=derived_sha1, filename=derived.name)
    try:
        assert client.exists(derived_sha1) == media_key
    finally:
        client.trash([derived_sha1])
    assert (still.stat().st_mtime_ns, video.stat().st_mtime_ns) == before
