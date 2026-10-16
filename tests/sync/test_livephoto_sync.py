"""Live Photo end to end: still + `.MOV` in, one Motion Photo out."""

import base64
import hashlib
from dataclasses import replace

from immich_gphotos.clock import FakeClock
from immich_gphotos.config import Filters, RetryPolicy
from immich_gphotos.gphotos.fake import FakeGooglePhotosClient
from immich_gphotos.immich.fake import FakeImmichClient
from immich_gphotos.models import Asset, AssetState, Outcome, Priority
from immich_gphotos.store.assets import AssetRepo
from immich_gphotos.store.db import connect
from immich_gphotos.sync.bytes import ByteResolver
from immich_gphotos.sync.livephoto import LIVE_PHOTO_PAIRED_REASON
from immich_gphotos.sync.worker import Worker

STILL_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 64 + b"\xff\xd9"
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128

STILL = Asset(
    immich_id="still-1",
    checksum="sum-still",
    filename="IMG_1.JPG",
    type="IMAGE",
    size_bytes=len(STILL_BYTES),
    immich_updated_at="2026-09-17T10:00:00Z",
    taken_at="2022-07-05T12:00:00.000Z",
    live_photo_video_id="mov-1",
    original_path=None,
    visibility="timeline",
    is_offline=False,
    is_trashed=False,
)

MOV = Asset(
    immich_id="mov-1",
    checksum="sum-mov",
    filename="IMG_1.MOV",
    type="VIDEO",
    size_bytes=len(VIDEO_BYTES),
    immich_updated_at="2026-09-17T10:00:00Z",
    taken_at="2022-07-05T12:00:00.000Z",
    live_photo_video_id=None,
    original_path=None,
    visibility="hidden",
    is_offline=False,
    is_trashed=False,
)


def _muxed_hash() -> str:
    from immich_gphotos.sync.livephoto import build_motion_photo_jpeg

    muxed = build_motion_photo_jpeg(STILL_BYTES, VIDEO_BYTES)
    return base64.b64encode(hashlib.sha1(muxed).digest()).decode()


def _rig(tmp_path):
    clock = FakeClock()
    assets = AssetRepo(connect(tmp_path / "t.db"), clock)
    immich = FakeImmichClient(contents={"still-1": STILL_BYTES, "mov-1": VIDEO_BYTES})
    gphotos = FakeGooglePhotosClient()
    resolver = ByteResolver(immich, scratch=tmp_path / "scratch", assets=assets)
    worker = Worker(assets, gphotos, resolver, Filters(), RetryPolicy(jitter=0.0), clock)
    return worker, assets, immich, gphotos


def _enqueue(assets: AssetRepo):
    assets.upsert_pending(STILL, Priority.WEBHOOK)
    assets.upsert_pending(MOV, Priority.WEBHOOK)


def test_resolve_muxes_the_pair_into_one_motion_photo(tmp_path):
    worker, assets, _, _ = _rig(tmp_path)
    _enqueue(assets)
    _ = worker

    resolver = ByteResolver(
        FakeImmichClient(contents={"still-1": STILL_BYTES, "mov-1": VIDEO_BYTES}),
        scratch=tmp_path / "scratch2",
        allow_direct=False,
        assets=assets,
    )
    resolved = resolver.resolve(STILL)

    assert resolved.temporary is True
    assert resolved.paired_video_id == "mov-1"
    data = resolved.path.read_bytes()
    assert data.startswith(b"\xff\xd8")
    assert data.endswith(VIDEO_BYTES)
    assert b"Camera:MotionPhoto" in data
    resolver.release(resolved)
    assert not resolved.path.exists()


def test_worker_uploads_one_muxed_file_and_pairs_the_clip(tmp_path):
    worker, assets, _, gphotos = _rig(tmp_path)
    _enqueue(assets)
    stored = assets.claim_next(limit=1)[0]
    assert stored.asset.immich_id == "still-1"  # webhook priority ties broken by first_seen
    result = worker.process(stored)

    assert result.state is AssetState.SYNCED
    assert result.outcome is Outcome.UPLOADED
    # One upload, described by the muxed bytes' hash -- not the still's.
    assert len(gphotos.uploads) == 1
    assert gphotos.uploads[0][0] == _muxed_hash()
    assert assets.get("mov-1").ineligible_reason == LIVE_PHOTO_PAIRED_REASON
    assert assets.get("still-1").media_key == result.media_key


def test_missing_companion_falls_back_to_the_still_alone(tmp_path):
    worker, assets, _, gphotos = _rig(tmp_path)
    assets.upsert_pending(STILL, Priority.WEBHOOK)
    result = worker.process(assets.claim_next(limit=1)[0])

    assert result.state is AssetState.SYNCED
    assert gphotos.uploads == [("sum-still", "IMG_1.JPG")]


def test_heic_still_falls_back_without_pairing(tmp_path):
    heic = STILL_BYTES.replace(b"\xff\xd8", b"\x00\x00", 1)
    clock = FakeClock()
    assets = AssetRepo(connect(tmp_path / "t.db"), clock)
    immich = FakeImmichClient(contents={"still-1": heic, "mov-1": VIDEO_BYTES})
    gphotos = FakeGooglePhotosClient()
    resolver = ByteResolver(immich, scratch=tmp_path / "scratch", assets=assets)
    worker = Worker(assets, gphotos, resolver, Filters(), RetryPolicy(jitter=0.0), clock)
    _enqueue(assets)

    result = worker.process(assets.claim_next(limit=1)[0])

    assert result.state is AssetState.SYNCED
    assert gphotos.uploads == [("sum-still", "IMG_1.JPG")]
    assert assets.get("mov-1").ineligible_reason != LIVE_PHOTO_PAIRED_REASON


def test_muxed_remote_hit_marks_synced_without_uploading(tmp_path):
    worker, assets, _, gphotos = _rig(tmp_path)
    _enqueue(assets)
    gphotos.present[_muxed_hash()] = "motion-key"

    result = worker.process(assets.claim_next(limit=1)[0])

    assert result.outcome is Outcome.ALREADY_PRESENT
    assert result.media_key == "motion-key"
    assert gphotos.uploads == []
    assert assets.get("mov-1").ineligible_reason == LIVE_PHOTO_PAIRED_REASON


def test_direct_read_pair_is_muxed_without_touching_originals(tmp_path):
    still_path = tmp_path / "IMG_1.JPG"
    mov_path = tmp_path / "IMG_1.MOV"
    still_path.write_bytes(STILL_BYTES)
    mov_path.write_bytes(VIDEO_BYTES)
    clock = FakeClock()
    assets = AssetRepo(connect(tmp_path / "t.db"), clock)
    immich = FakeImmichClient()
    resolver = ByteResolver(immich, scratch=tmp_path / "scratch", assets=assets)
    assets.upsert_pending(replace(STILL, original_path=str(still_path)), Priority.WEBHOOK)
    assets.upsert_pending(replace(MOV, original_path=str(mov_path)), Priority.WEBHOOK)

    resolved = resolver.resolve(replace(STILL, original_path=str(still_path)))

    assert resolved.paired_video_id == "mov-1"
    assert resolved.path.read_bytes().endswith(VIDEO_BYTES)
    assert still_path.read_bytes() == STILL_BYTES  # untouched
    assert mov_path.read_bytes() == VIDEO_BYTES  # untouched
    resolver.release(resolved)
