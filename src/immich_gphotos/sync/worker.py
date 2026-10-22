import logging
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from immich_gphotos.clock import Clock
from immich_gphotos.config import Filters, RetryPolicy
from immich_gphotos.gphotos.protocol import GooglePhotosClient, GPhotosError
from immich_gphotos.models import Asset, AssetState, ErrorClass, Outcome, StoredAsset
from immich_gphotos.store.assets import AssetRepo
from immich_gphotos.sync.backoff import next_delay, should_quarantine
from immich_gphotos.sync.bytes import ByteResolver, SourceChangedError
from immich_gphotos.sync.eligibility import check_eligibility
from immich_gphotos.sync.favorites import FavoriteSyncer
from immich_gphotos.sync.livephoto import LIVE_PHOTO_PAIRED_REASON
from immich_gphotos.sync.throttle import TokenBucket

HALT_RETRY_DELAY = timedelta(minutes=10)

# How long a transfer-window-deferred asset waits before being reclaimed. Long
# enough that an idle background loop (which ticks every couple of seconds)
# does not re-run the remote hash check dozens of times a minute for the
# whole closed window; short enough that transfer resumes promptly once the
# window reopens or is widened.
WINDOW_RETRY_DELAY = timedelta(minutes=15)

# The most `_throttle_upload` will ever block the calling thread for. Below
# this, the wait is short enough that sleeping right here -- on whatever
# thread called `process()`, which for the single background-loop tick is
# the *only* thread driving reconcile, backfill, album sync, the deletion
# sweep, requeue_stale_uploading and the pause-retry -- is a tolerable,
# bounded delay. At or above it, that same thread would otherwise be frozen
# for as long as the configured bandwidth cap and this file's size imply --
# up to hours at the documented minimum, `MIN_BANDWIDTH_BYTES_PER_SECOND` in
# `api.routes` -- so the wait is paid by requeuing the asset for `now + wait`
# instead (see the `deferred` branch in `_throttle_upload`), the same
# mechanism the schedule-window path above already uses. 30s is well under
# every other timer in this module (WINDOW_RETRY_DELAY, HALT_RETRY_DELAY)
# and a small multiple of IDLE_SLEEP_SECONDS, so it costs at most a handful
# of idle ticks' worth of responsiveness for any wait that clears it.
MAX_INLINE_THROTTLE_WAIT_SECONDS = 30.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerResult:
    state: AssetState
    outcome: Outcome | None = None
    media_key: str | None = None
    reason: str | None = None
    error_class: ErrorClass | None = None
    halt: bool = False
    deferred: bool = False


class Worker:
    """Process one asset. Every step is idempotent, so re-running is always safe."""

    def __init__(
        self,
        assets: AssetRepo,
        gphotos: GooglePhotosClient,
        resolver: ByteResolver,
        filters: Filters,
        retry: RetryPolicy,
        clock: Clock,
        *,
        bandwidth: TokenBucket | None = None,
        gate: AbstractContextManager[Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        favorites_enabled: bool = True,
        favorites: FavoriteSyncer | None = None,
    ) -> None:
        self._assets = assets
        self._gphotos = gphotos
        self._resolver = resolver
        self._filters = filters
        self._retry = retry
        self._clock = clock
        # Shared across every worker thread that calls this same Worker
        # instance's process() -- one bucket, not one per thread, or the cap
        # would be wrong by a factor of the pool size. None means no cap and
        # must add no overhead (see _throttle_upload below).
        self._bandwidth = bandwidth
        # The process-wide upload gate (Task 6): one instance -- production
        # passes a `threading.Semaphore` sized to the global `worker_threads`
        # -- shared by every account's Worker, so N accounts' pools cannot
        # multiply the effective upload concurrency by N. Any context
        # manager works; None (the default) means no gate and costs nothing
        # extra. See the acquire site in `process` for exactly what it wraps
        # and why.
        self._gate = gate
        # Injectable so tests can drive the wait with a FakeClock and assert
        # on it without a real sleep; production uses the real time.sleep.
        self._sleep = sleep
        # Whether a favorited asset is starred in Google immediately after its
        # bytes land (or dedup clears it). Off means the worker never calls
        # set_favorite and every drift waits for the sweeper -- which itself
        # is gated on the same flag in `BackgroundLoops`, so off is fully off.
        self._favorites_enabled = favorites_enabled
        # The per-item favorite reconciler shared with the sweeper, so the
        # immediate star and the periodic sweep serialise on the same item
        # lock and read/write the same per-item ledger.
        self._favorites = favorites if favorites is not None else FavoriteSyncer(gphotos, assets)
        # Ids for which a previous call already charged the bucket (via
        # `TokenBucket.take`) but deferred rather than sleeping out the
        # result -- see `_throttle_upload`. When such an asset is reclaimed,
        # metering it again would charge the same bytes against the bucket
        # twice for a transfer that has only ever happened once; since the
        # bucket's capacity never exceeds one second's worth of tokens, that
        # second charge would recreate nearly the same wait on every
        # subsequent retry and the asset would never actually upload. This
        # is in-memory only, same as the bucket itself -- lost on restart,
        # which just re-meters the asset fresh from a newly-full bucket
        # rather than double-charging a debt no longer tracked, a fine
        # degradation and not a correctness bug. An entry lives only as long
        # as its asset is genuinely still in flight: `process` drops it on
        # every exit except the two deliberate defer returns, so an asset
        # that goes terminal (or fails) between the defer and the retry does
        # not leave its id here for the life of the process.
        self._throttle_prepaid: set[str] = set()
        self._throttle_prepaid_lock = threading.Lock()

    def process(
        self,
        stored: StoredAsset,
        *,
        transfer_allowed: bool = True,
        album_allowlist_ids: frozenset[str] | None = None,
    ) -> WorkerResult:
        """Process one asset.

        `transfer_allowed` gates only the byte-moving step (resolving bytes
        off Immich and uploading them to Google). Eligibility, the local
        duplicate lookup and the remote hash check all run regardless, since
        the spec calls those "tiny" and wants them clearing the queue for
        free at any hour. An asset that clears eligibility and both dedup
        checks but still needs bytes moved, and is asked outside the
        schedule window, is handed back to the queue rather than uploaded --
        see the `deferred` branch below.

        `album_allowlist_ids` is the set of Immich asset ids the runtime
        resolved (once per tick, from `filters.album_allowlist`) as
        belonging to an allowed album. It is only consulted when
        `filters.album_allowlist` is actually set.
        """
        asset = stored.asset
        # `_throttle_prepaid` must only ever hold assets that are genuinely
        # still in flight -- i.e. ones this process deliberately handed back
        # to the queue to wait out a charge the bucket has already taken.
        # Every other way out of this call ends that transfer's life for
        # now, so the entry is dropped: `_throttle_upload` is skipped
        # entirely by the eligibility, local-dedup and remote-dedup returns
        # and by any exception, and an asset that lands in a terminal state
        # there (trashed between the defer and the retry, say) is never
        # claimed again, so nothing would ever come back to clear it. Hence
        # the flag rather than a blanket `finally`: the two defer returns
        # are exactly the paths that must keep it.
        retain_prepaid = False
        try:
            reason = check_eligibility(asset, self._filters, album_allowlist_ids)
            if reason is not None:
                self._assets.mark_ineligible(asset.immich_id, reason)
                return WorkerResult(AssetState.INELIGIBLE, reason=reason)

            local_key = self._assets.media_key_for_checksum(asset.checksum)
            if local_key:
                self._assets.mark_synced(asset.immich_id, local_key, Outcome.ALREADY_PRESENT)
                self._star_if_favorited(asset, local_key)
                return WorkerResult(AssetState.SYNCED, Outcome.ALREADY_PRESENT, local_key)

            try:
                # Immich's checksum keyed against Google's content-hash
                # index: exact only for Internal uploads (whose checksum IS
                # the content hash), where it can clear an asset without
                # ever resolving bytes. External Library checksums are
                # sha1(path) and lookups on them reliably miss -- the
                # authoritative content-hash check runs right after resolve,
                # when the real bytes' hash is available.
                remote_key = self._gphotos.exists(asset.checksum)
                if remote_key:
                    self._assets.mark_synced(asset.immich_id, remote_key, Outcome.ALREADY_PRESENT)
                    self._star_if_favorited(asset, remote_key)
                    return WorkerResult(AssetState.SYNCED, Outcome.ALREADY_PRESENT, remote_key)
                if not transfer_allowed:
                    # Neither dedup check cleared it for free -- this asset
                    # genuinely needs bytes moved. That's the one thing the
                    # schedule window gates. Hand it back to PENDING without
                    # counting an attempt (this is not the asset's fault, the
                    # same reasoning `requeue` already exists for on the halt
                    # path) and move on; the runtime keeps processing the rest
                    # of the batch rather than stopping.
                    #
                    # Still in flight, so any charge already taken for it
                    # stands and the retry inside the window must not be
                    # metered a second time -- but only once the requeue has
                    # actually happened. If it raises (a locked database, a
                    # full disk), this asset is *not* deferred: the exception
                    # goes to `_handle_failure` below, and a flag set ahead of
                    # the call would leave a stale id in `_throttle_prepaid`
                    # for an asset with nothing in flight, so its next
                    # successful attempt would skip metering entirely.
                    self._assets.requeue(asset.immich_id, self._clock.now() + WINDOW_RETRY_DELAY)
                    retain_prepaid = True
                    return WorkerResult(
                        AssetState.PENDING, reason="deferred: outside the transfer window", deferred=True
                    )

                resolved = self._resolver.resolve(asset)
                try:
                    # The resolver certified the SHA-1 of these exact bytes.
                    # That is the hash every Google-side consumer keys on --
                    # dedup lookup, upload integrity, trash -- and Immich's
                    # own checksum is that hash only for Internal uploads
                    # (External Library rows carry a virtual sha1(path)
                    # checksum that does not describe any real bytes). One
                    # hash lookup against the resolved bytes covers ordinary
                    # uploads and muxed Motion Photos alike.
                    content_hash = resolved.sha1_b64
                    assert content_hash is not None  # resolve() always certifies the bytes
                    content_remote: str | None
                    try:
                        content_remote = self._gphotos.exists(content_hash)
                    except GPhotosError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - a dedup miss just uploads
                        logger.debug("content hash lookup failed for %s: %s", asset.immich_id, exc)
                        content_remote = None
                    if content_remote:
                        self._assets.mark_synced(
                            asset.immich_id,
                            content_remote,
                            Outcome.ALREADY_PRESENT,
                            upload_sha1=content_hash,
                            notes=resolved.notes,
                        )
                        if resolved.paired_video_id is not None:
                            self._mark_paired(resolved.paired_video_id)
                        self._star_if_favorited(asset, content_remote)
                        return WorkerResult(AssetState.SYNCED, Outcome.ALREADY_PRESENT, content_remote)

                    defer_wait = self._throttle_upload(asset.immich_id, resolved.path)
                    if defer_wait is not None:
                        # The bandwidth cap would otherwise block this thread
                        # for longer than MAX_INLINE_THROTTLE_WAIT_SECONDS --
                        # see `_throttle_upload`. Hand it back to the queue
                        # instead of sleeping on it, the same way the
                        # schedule-window branch above does; not the asset's
                        # fault, so no attempt is burned. The bucket has
                        # already been charged for this transfer
                        # (`_throttle_prepaid` remembers that, and this is the
                        # path that must keep it), so the retry that reclaims
                        # it after `defer_wait` must not be metered again --
                        # but only once the requeue has actually happened, for
                        # the same reason as the window branch above.
                        #
                        # `defer_for_bandwidth` rather than `requeue`: the
                        # deadline below is arithmetic against the *current*
                        # bandwidth cap, and is meaningless once that cap
                        # changes, so the row is marked as such and a cap
                        # change releases it (see
                        # `AssetRepo.release_bandwidth_deferrals`).
                        self._assets.defer_for_bandwidth(
                            asset.immich_id, self._clock.now() + timedelta(seconds=defer_wait)
                        )
                        retain_prepaid = True
                        return WorkerResult(
                            AssetState.PENDING,
                            reason="deferred: bandwidth cap would block the loop too long",
                            deferred=True,
                        )
                    # The gate wraps this call alone -- never the wider
                    # resolve/throttle/upload region above it. `ByteResolver.
                    # resolve` runs *before* `_throttle_upload` (metering
                    # needs the resolved file's size), and a wait below
                    # MAX_INLINE_THROTTLE_WAIT_SECONDS is slept out right here
                    # on this thread (see `_throttle_upload`'s `defer_wait is
                    # None` branch after an inline sleep). Wrapping the gate
                    # around that region too would hold the process-wide
                    # upload slot for the whole sleep -- for a transfer that
                    # has not even started moving bytes to Google yet -- and
                    # reintroduce exactly the cross-account stalling that
                    # giving each account its own worker thread (Tasks 1-5)
                    # exists to remove. The upload call is the only thing
                    # here that actually touches the shared WAN resource the
                    # gate protects, so it is the only thing the gate needs
                    # to bound. This is precisely the line a future reader
                    # will try to "tidy" outward -- don't.
                    if self._gate is not None:
                        with self._gate:
                            media_key = self._gphotos.upload(
                                resolved.path, checksum=content_hash, filename=asset.filename
                            )
                    else:
                        media_key = self._gphotos.upload(
                            resolved.path, checksum=content_hash, filename=asset.filename
                        )
                    if not self._resolver.still_unchanged(resolved):
                        # A direct-read original was rewritten while gpmc was
                        # streaming it: what Google received may not be the
                        # bytes `content_hash` describes. Never record it under
                        # that hash; retry, which re-hashes and re-checks
                        # Google read-only before any further upload.
                        raise SourceChangedError(f"{asset.filename} changed during upload")
                finally:
                    self._resolver.release(resolved)

                self._assets.mark_synced(
                    asset.immich_id,
                    media_key,
                    Outcome.UPLOADED,
                    upload_sha1=content_hash,
                    notes=resolved.notes,
                )
                self._star_if_favorited(asset, media_key)
                if resolved.paired_video_id is not None:
                    self._mark_paired(resolved.paired_video_id)
                return WorkerResult(AssetState.SYNCED, Outcome.UPLOADED, media_key)

            except GPhotosError as exc:
                return self._handle_failure(stored, exc.error_class, str(exc))
            except OSError as exc:
                return self._handle_failure(stored, ErrorClass.ASSET_UNAVAILABLE, str(exc))
            except Exception as exc:  # noqa: BLE001 - never let one asset kill the worker
                # last_error only keeps a 500-char truncated message with no traceback,
                # so the full exception must be captured here or it is lost forever.
                logger.exception("unexpected error processing asset %s", asset.immich_id)
                return self._handle_failure(stored, ErrorClass.UNKNOWN, str(exc))
        finally:
            # No bucket configured means nothing was ever charged and the
            # set is always empty, so the uncapped path stays lock-free.
            if not retain_prepaid and self._bandwidth is not None:
                with self._throttle_prepaid_lock:
                    self._throttle_prepaid.discard(asset.immich_id)

    def _star_if_favorited(self, asset: Asset, media_key: str) -> None:
        """Star a newly-synced asset in Google when Immich has it starred.

        Best-effort by design: the bytes are already safe (mark_synced ran),
        so a star that fails -- or a database write that fails -- leaves the
        row drifted and the sweeper retries it next tick. Never raises, never
        halts transfer, never burns an upload attempt. Only True is pushed
        here; un-starring (and any True that failed here) is the sweeper's
        job, which is also what covers stars added after the bytes synced.

        Disabled entirely when `favorites_enabled` is off.
        """
        if not self._favorites_enabled or not asset.is_favorite:
            return
        try:
            # Per Google item, merged across every row that dedups onto it --
            # see FavoriteSyncer. Starring is always safe to do immediately.
            self._favorites.reconcile_key(media_key)
        except Exception as exc:  # noqa: BLE001 - a star is never worth failing a backup over
            logger.debug("could not star %s in Google Photos: %s", asset.immich_id, exc)

    def _mark_paired(self, companion_id: str) -> None:
        """Account for a companion `.MOV` folded into its still's upload.

        Best-effort: the still is already SYNCED, so a bookkeeping write
        that fails just leaves the companion `hidden` for now rather than
        failing anything. Overwrites whatever reason the companion carried
        (`hidden` when the worker saw it first, or still PENDING when the
        still won the race) -- both are terminal-or-harmless from here.
        """
        try:
            self._assets.mark_ineligible(companion_id, LIVE_PHOTO_PAIRED_REASON)
        except Exception as exc:  # noqa: BLE001 - bookkeeping, never worth failing over
            logger.debug("could not mark Live Photo motion %s paired: %s", companion_id, exc)

    def _throttle_upload(self, immich_id: str, path: Path) -> float | None:
        """Make the caller wait its share of the configured upload bandwidth cap,
        or say how long the *asset* should wait instead, when that share is
        too long for this thread to sleep on.

        Only the upload leg is metered, not the download/resolve leg: the cap
        is documented (and read back from the API) as "cap on upload
        throughput" -- it protects the outbound link to Google, which is
        often the metered/constrained one (e.g. a residential or mobile
        upstream), not the local read from Immich's own volume or API, which
        is a different link entirely and, in the direct-read fast path, may
        not even cross the network. Metering it too would throttle local disk
        reads for no reason the setting claims to cover.

        No bucket configured (`self._bandwidth is None`, i.e. no cap) takes
        this branch and returns immediately -- no lock, no clock call, no
        stat() -- so an unconfigured cap adds no overhead.

        Returns `None` when the caller may proceed with the upload right
        away (no cap, a wait it already slept out inline, or an asset whose
        debt was already paid by an earlier call -- see `_throttle_prepaid`
        on `__init__`). Returns the number of seconds the caller should defer
        the asset by instead of sleeping, when the computed wait is at or
        above MAX_INLINE_THROTTLE_WAIT_SECONDS.
        """
        if self._bandwidth is None:
            return None
        with self._throttle_prepaid_lock:
            if immich_id in self._throttle_prepaid:
                # A previous call already charged the bucket for this exact
                # transfer and deferred rather than sleeping; charging it
                # again here would double-count the same bytes. Upload now,
                # unmetered a second time -- the real wait already elapsed
                # while this asset sat requeued.
                self._throttle_prepaid.discard(immich_id)
                return None
        try:
            size = path.stat().st_size
        except OSError:
            # Can't size the file (already gone, races with release, ...) --
            # never let metering itself fail the upload.
            return None
        wait = self._bandwidth.take(size)
        if wait < MAX_INLINE_THROTTLE_WAIT_SECONDS:
            if wait > 0:
                self._sleep(wait)
            return None
        with self._throttle_prepaid_lock:
            self._throttle_prepaid.add(immich_id)
        return wait

    def _handle_failure(self, stored: StoredAsset, error_class: ErrorClass, message: str) -> WorkerResult:
        asset_id = stored.asset.immich_id

        if error_class.halts_transfer():
            self._assets.requeue(asset_id, self._clock.now() + HALT_RETRY_DELAY)
            return WorkerResult(AssetState.PENDING, error_class=error_class, halt=True)

        attempts = stored.attempts + 1
        if should_quarantine(attempts, self._retry):
            self._assets.mark_failed(asset_id, error_class, message)
            return WorkerResult(AssetState.FAILED, error_class=error_class, reason=message)

        delay = next_delay(attempts, self._retry)
        self._assets.mark_retry(asset_id, error_class, message, self._clock.now() + delay)
        return WorkerResult(AssetState.PENDING, error_class=error_class, reason=message)
