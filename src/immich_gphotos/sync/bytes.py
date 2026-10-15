import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from immich_gphotos.immich.protocol import ImmichClient
from immich_gphotos.models import Asset

logger = logging.getLogger(__name__)


def _stamp_capture_date(path: Path, taken_at: str | None) -> None:
    """Set `path`'s mtime to when the asset was actually taken.

    gpmc uploads a file's `st_mtime` to Google as the capture timestamp, and
    Google honours it for any file whose bytes carry no date of their own --
    WhatsApp videos, screenshots, anything stripped of EXIF. Immich already
    stores its own originals with the capture date as the mtime, so the
    direct-read path was always correct; a file we download through the API
    lands with an mtime of "now", which is what put those assets in Google
    Photos dated today instead of when they were taken.

    Best-effort by design: a missing, malformed or unrepresentable date leaves
    the file alone and the asset still uploads, dated by Google as before. A
    timestamp is never worth failing a backup over.
    """
    if not taken_at:
        return
    try:
        when = datetime.fromisoformat(taken_at.replace("Z", "+00:00")).timestamp()
        os.utime(path, (when, when))
    except (ValueError, OSError, OverflowError) as exc:
        logger.debug("could not stamp capture date %r onto %s: %s", taken_at, path.name, exc)


@dataclass(frozen=True)
class ResolvedBytes:
    path: Path
    temporary: bool
    # When the still was muxed with its Live Photo motion before upload,
    # the Immich id of the companion `.MOV` that was folded in. The worker
    # marks that row `live_photo_paired` after the upload lands, so the
    # hidden clip is accounted for instead of lingering as `hidden`. None
    # for every ordinary (unmuxed) resolution.
    paired_video_id: str | None = None


class ByteResolver:
    """Get at an asset's bytes by the cheapest route available.

    When the container can see Immich's library volume, `originalPath` is handed
    straight to gpmc: no copy, no scratch space, no traffic through the API.
    Otherwise the original is downloaded to scratch and deleted afterwards.

    An Apple Live Photo still (one carrying `live_photo_video_id`) is muxed
    with its companion `.MOV` into a single Google Motion Photo before it is
    handed back -- see `_try_resolve_live_photo`. Anything that makes muxing
    impossible (companion row not yet in the database, bytes unreadable, still
    not a JPEG, oversized or unrecognised video) falls back to the still
    alone, as before: motion is never worth failing a backup over.
    """

    def __init__(
        self,
        immich: ImmichClient,
        scratch: Path,
        allow_direct: bool = True,
        assets=None,  # noqa: ANN001 - AssetRepo; only for the Live Photo companion lookup
    ) -> None:
        self._immich = immich
        self._scratch = Path(scratch)
        self._allow_direct = allow_direct
        self._assets = assets

    def resolve(self, asset: Asset) -> ResolvedBytes:
        if asset.live_photo_video_id and self._assets is not None:
            muxed = self._try_resolve_live_photo(asset)
            if muxed is not None:
                return muxed
        return self._resolve_single(asset)

    def _scratch_path(self, asset: Asset) -> Path:
        """A fresh scratch path for one resolution.

        The nonce guarantees a fresh path per call: two concurrent
        resolutions of the *same* asset (e.g. a slow-but-healthy upload that
        crosses requeue_stale_uploading's liveness-free age threshold and gets
        handed to a second worker) must never share a scratch file, or one
        caller's release() could delete bytes the other is still writing or
        uploading. Only the basename of the filename is used because
        `originalFileName` arrives unsanitized from Immich.
        """
        self._scratch.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        return self._scratch / f"{asset.immich_id}-{nonce}-{Path(asset.filename).name}"

    def _try_resolve_live_photo(self, still: Asset) -> ResolvedBytes | None:
        """Mux a Live Photo still with its companion motion, or None.

        None means "upload the still alone": the companion row is missing,
        either side's bytes cannot be read, the still is not a JPEG (HEIC
        Motion Photos need an `mpvd` box plus `Padding`, a separate project),
        or the video is empty/oversized/unrecognised. Every such case logs
        at debug and falls back -- never raises out of `resolve`.
        """
        from immich_gphotos.sync import livephoto

        assert self._assets is not None
        try:
            companion_stored = self._assets.get(still.live_photo_video_id or "")
        except Exception as exc:  # noqa: BLE001 - a lookup failure costs motion, not the backup
            logger.debug("could not look up Live Photo motion for %s: %s", still.immich_id, exc)
            return None
        if companion_stored is None:
            return None
        companion = companion_stored.asset
        try:
            still_single = self._resolve_single(still)
        except Exception as exc:  # noqa: BLE001 - motion is never worth failing the still over
            logger.debug("could not resolve Live Photo still %s: %s", still.immich_id, exc)
            return None
        try:
            companion_single = self._resolve_single(companion)
        except Exception as exc:  # noqa: BLE001 - see above
            logger.debug("could not resolve Live Photo motion for %s: %s", still.immich_id, exc)
            self.release(still_single)
            return None
        try:
            try:
                still_bytes = still_single.path.read_bytes()
                video_bytes = companion_single.path.read_bytes()
            except OSError as exc:
                logger.debug("could not read Live Photo pair for %s: %s", still.immich_id, exc)
                return None
            try:
                muxed = livephoto.build_motion_photo_jpeg(
                    still_bytes,
                    video_bytes,
                    livephoto.video_mime_for_filename(companion.filename),
                )
            except ValueError as exc:
                logger.debug("not muxing Live Photo %s: %s", still.immich_id, exc)
                return None
            muxed_path = self._scratch_path(still)
            try:
                muxed_path.write_bytes(muxed)
            except OSError as exc:
                logger.debug("could not stage muxed Live Photo %s: %s", still.immich_id, exc)
                muxed_path.unlink(missing_ok=True)
                return None
            _stamp_capture_date(muxed_path, still.taken_at or self._lookup_taken_at(still))
            return ResolvedBytes(path=muxed_path, temporary=True, paired_video_id=companion.immich_id)
        finally:
            # The muxed file is an independent copy; both source files are
            # no longer needed once it is staged (or once we have given up).
            self.release(still_single)
            self.release(companion_single)

    def _resolve_single(self, asset: Asset) -> ResolvedBytes:
        if self._allow_direct and asset.original_path:
            candidate = Path(asset.original_path)
            if candidate.is_file():
                return ResolvedBytes(path=candidate, temporary=False)
        dest = self._scratch_path(asset)
        self._immich.download_original(asset.immich_id, dest)
        # Only ever the scratch copy: Immich's own originals already carry the
        # right mtime and its library is mounted read-only.
        _stamp_capture_date(dest, asset.taken_at or self._lookup_taken_at(asset))
        return ResolvedBytes(path=dest, temporary=True)

    def _lookup_taken_at(self, asset: Asset) -> str | None:
        """Ask Immich for a capture date the stored row does not have.

        Only reached on the download path, and only for a row whose `taken_at`
        is empty -- in practice one queued before that column existed. Those
        rows are never re-read before the worker claims them (a reconcile or
        backfill pass only refreshes what it happens to walk), so without this
        the entire backlog present at upgrade time would upload dated today.
        Steady-state rows already carry the date and never get here, so this
        costs one extra request per legacy asset, once, against a download
        that was already far more expensive.

        Swallows everything: the bytes are on disk and the upload is the
        valuable part, so a metadata call that fails costs the date, not the
        backup.
        """
        try:
            return self._immich.asset_taken_at(asset.immich_id)
        except Exception as exc:  # noqa: BLE001 - a date is never worth failing a backup over
            logger.debug("could not look up the capture date for %s: %s", asset.immich_id, exc)
            return None

    def release(self, resolved: ResolvedBytes) -> None:
        if resolved.temporary:
            resolved.path.unlink(missing_ok=True)
