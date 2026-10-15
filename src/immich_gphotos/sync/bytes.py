import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from immich_gphotos.immich.protocol import ImmichClient
from immich_gphotos.models import Asset

logger = logging.getLogger(__name__)

# A direct-read original whose mtime is within this of `taken_at` is handed
# straight to gpmc. Anything further off is a file whose mtime no longer
# reflects when the photo was taken -- a NAS move/copy/import that reset it
# to the import date is the common case -- and gpmc would send that wrong
# mtime to Google as the capture timestamp for any bytes carrying no date of
# their own. 2s covers filesystem granularity and second-truncated dates
# without copying libraries whose mtimes are merely rounded.
MTIME_TOLERANCE_SECONDS = 2.0


def _parse_taken_at(taken_at: str | None) -> float | None:
    """Parse Immich's `fileCreatedAt` to a unix timestamp, or None.

    None covers every "no usable date" shape: missing, empty, malformed or
    unrepresentable. Callers treat None as "cannot verify the mtime, upload
    as before" -- a timestamp is never worth failing a backup over.
    """
    if not taken_at:
        return None
    try:
        return datetime.fromisoformat(taken_at.replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return None


def _stamp_capture_date(path: Path, taken_at: str | None) -> None:
    """Set `path`'s mtime to when the asset was actually taken.

    gpmc uploads a file's `st_mtime` to Google as the capture timestamp, and
    Google honours it for any file whose bytes carry no date of their own --
    WhatsApp videos, screenshots, anything stripped of EXIF. A file downloaded
    through the API lands with an mtime of "now", which is what put those
    assets in Google Photos dated today instead of when they were taken.

    Best-effort by design: a missing, malformed or unrepresentable date leaves
    the file alone and the asset still uploads, dated by Google as before. A
    timestamp is never worth failing a backup over.
    """
    when = _parse_taken_at(taken_at)
    if when is None:
        return
    try:
        os.utime(path, (when, when))
    except (OSError, OverflowError) as exc:
        logger.debug("could not stamp capture date %r onto %s: %s", taken_at, path.name, exc)


@dataclass(frozen=True)
class ResolvedBytes:
    path: Path
    temporary: bool


class ByteResolver:
    """Get at an asset's bytes by the cheapest route available.

    When the container can see Immich's library volume, `originalPath` is
    handed straight to gpmc -- but only when that file's mtime still reflects
    when the photo was taken. Immich normally stores its originals with the
    capture date as the mtime, and then the direct path is a zero-copy win;
    a move, copy or re-import on some NAS/server setups resets the mtime to
    the import date instead, and gpmc would send that wrong mtime to Google
    as the capture timestamp for any bytes carrying no date of their own.
    A direct file whose mtime disagrees with `taken_at` is therefore copied
    to scratch, stamped, and uploaded from there -- Immich's own file is
    never modified (its library is mounted read-only) and never re-dated.

    Otherwise the original is downloaded to scratch and deleted afterwards.
    """

    def __init__(self, immich: ImmichClient, scratch: Path, allow_direct: bool = True) -> None:
        self._immich = immich
        self._scratch = Path(scratch)
        self._allow_direct = allow_direct

    def _scratch_path(self, asset: Asset) -> Path:
        """A fresh scratch path for one resolution.

        The nonce guarantees a fresh path per call: two concurrent
        resolutions of the *same* asset (e.g. a slow-but-healthy upload that
        crosses requeue_stale_uploading's liveness-free age threshold and gets
        handed to a second worker) must never share a scratch file, or one
        caller's release() could delete bytes the other is still writing or
        uploading. Only the basename of the filename is used because
        `originalFileName` arrives unsanitized from Immich, and a separator
        in it must not turn into a subdirectory.
        """
        self._scratch.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        return self._scratch / f"{asset.immich_id}-{nonce}-{Path(asset.filename).name}"

    def _direct_or_copy(self, candidate: Path, asset: Asset, taken_at: str | None) -> ResolvedBytes:
        """Hand back `candidate` itself when its mtime already agrees with the
        capture date, else a stamped scratch copy of it.

        `taken_at` is the resolved capture-date string (already falling back
        to a lookup for legacy rows); None means there is no date to check
        against, so the file goes up as-is, as before.
        """
        when = _parse_taken_at(taken_at)
        if when is None:
            return ResolvedBytes(path=candidate, temporary=False)
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            # Gone or unreadable between the is_file() check and now: let the
            # caller fall through to the API-download path by signalling
            # "no direct file" with an exception the caller treats that way.
            # Simpler than returning a sentinel, and never fails the backup
            # on a race.
            raise
        if abs(mtime - when) <= MTIME_TOLERANCE_SECONDS:
            return ResolvedBytes(path=candidate, temporary=False)
        dest = self._scratch_path(asset)
        try:
            shutil.copyfile(candidate, dest)
        except OSError as exc:
            # The copy is an optimisation for correctness; losing it must
            # never lose the backup. Fall back to handing back the original:
            # the upload still happens, dated by Google as before.
            logger.debug("could not copy %s to scratch for re-dating: %s", candidate.name, exc)
            return ResolvedBytes(path=candidate, temporary=False)
        _stamp_capture_date(dest, taken_at)
        return ResolvedBytes(path=dest, temporary=True)

    def resolve(self, asset: Asset) -> ResolvedBytes:
        if self._allow_direct and asset.original_path:
            candidate = Path(asset.original_path)
            if candidate.is_file():
                try:
                    return self._direct_or_copy(
                        candidate, asset, asset.taken_at or self._lookup_taken_at(asset)
                    )
                except OSError:
                    pass
        self._scratch.mkdir(parents=True, exist_ok=True)
        dest = self._scratch_path(asset)
        self._immich.download_original(asset.immich_id, dest)
        # The scratch copy only: a direct-read original is either already
        # correct (handed back above) or was copied-and-stamped there, and
        # Immich's library is mounted read-only.
        _stamp_capture_date(dest, asset.taken_at or self._lookup_taken_at(asset))
        return ResolvedBytes(path=dest, temporary=True)

    def _lookup_taken_at(self, asset: Asset) -> str | None:
        """Ask Immich for a capture date the stored row does not have.

        Reached only for a row whose `taken_at` is empty -- in practice one
        queued before that column existed. Those rows are never re-read
        before the worker claims them (a reconcile or backfill pass only
        refreshes what it happens to walk), so without this the entire
        backlog present at upgrade time would upload dated today.
        Steady-state rows already carry the date and never get here, so this
        costs one extra request per legacy asset, once. On the download path
        that request runs against bytes already on disk; on the direct path
        it decides whether the original's mtime can be trusted at all.

        Swallows everything: the bytes are on disk (or the direct file is
        still uploadable) and the upload is the valuable part, so a metadata
        call that fails costs the date, not the backup.
        """
        try:
            return self._immich.asset_taken_at(asset.immich_id)
        except Exception as exc:  # noqa: BLE001 - a date is never worth failing a backup over
            logger.debug("could not look up the capture date for %s: %s", asset.immich_id, exc)
            return None

    def release(self, resolved: ResolvedBytes) -> None:
        if resolved.temporary:
            resolved.path.unlink(missing_ok=True)
