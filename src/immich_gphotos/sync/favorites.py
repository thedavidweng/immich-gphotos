import logging
from dataclasses import dataclass

from immich_gphotos.gphotos.protocol import GooglePhotosClient
from immich_gphotos.store.assets import AssetRepo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FavoriteSyncResult:
    synced: int = 0
    failed: int = 0


class FavoriteSyncer:
    """Push Immich star state to Google Photos.

    The worker stars newly-uploaded assets immediately (so a favorited photo
    is favorited in Google the moment it lands), but a star added or removed
    in Immich *after* the bytes synced never re-enters the upload queue --
    SYNCED is terminal and bytes don't move twice. This sweeper closes that
    gap: every reconcile tick it pushes wherever `is_favorite` (what Immich
    says now) disagrees with `favorite_synced` (what Google was last told).

    One failure never stops the sweep: the row is simply left drifted for the
    next tick, which retries it like any other failure. Favorites never halt
    transfer and never burn upload attempts -- the bytes are already safe.
    """

    def __init__(self, gphotos: GooglePhotosClient, assets: AssetRepo) -> None:
        self._gphotos = gphotos
        self._assets = assets

    def sync_once(self, limit: int = 500) -> FavoriteSyncResult:
        synced = failed = 0
        for stored in self._assets.favorite_drift(limit=limit):
            if not stored.media_key:
                continue
            try:
                self._gphotos.set_favorite(stored.media_key, stored.asset.is_favorite)
            except Exception as exc:  # noqa: BLE001 - one star must not kill the sweep
                failed += 1
                logger.debug("could not sync favorite for %s: %s", stored.asset.immich_id, exc)
                continue
            self._assets.mark_favorite_synced(stored.asset.immich_id, stored.asset.is_favorite)
            synced += 1
        return FavoriteSyncResult(synced=synced, failed=failed)
