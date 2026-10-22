from dataclasses import dataclass, field

from immich_gphotos.config import DeletionPolicy, Settings
from immich_gphotos.gphotos.protocol import GooglePhotosClient
from immich_gphotos.models import AssetState, StoredAsset
from immich_gphotos.store.assets import AssetRepo


def deletion_allowed(count: int, synced_total: int, policy: DeletionPolicy) -> tuple[bool, str | None]:
    """Refuse implausibly large deletions.

    The scenario guarded against is an Immich database restored from an old
    backup, with this service faithfully removing years of photos from Google.
    Neither bound can be disabled.
    """
    if count == 0:
        return True, None

    absolute_breach = count > policy.max_absolute
    # A non-positive synced_total with a non-zero count means there is no
    # library to measure a fraction against (or the count is nonsensical).
    # That must block rather than divide, so a caller that ever reports (or
    # miscomputes) an empty library cannot silently disable this bound.
    fraction_breach = synced_total <= 0 or (count / synced_total) > policy.max_fraction

    if not absolute_breach and not fraction_breach:
        return True, None

    reasons = []
    if absolute_breach:
        reasons.append(f"absolute limit exceeded: {count} > {policy.max_absolute}")
    if fraction_breach:
        if synced_total <= 0:
            reasons.append(f"fraction limit exceeded: {count} to delete against {synced_total} synced")
        else:
            reasons.append(f"fraction limit exceeded: {count}/{synced_total} > {policy.max_fraction:.0%}")
    return False, "; ".join(reasons)


@dataclass(frozen=True)
class DeletionPlan:
    # Content SHA-1s Google will be asked to trash (one per remote item).
    checksums: list[str] = field(default_factory=list)
    # Rows whose Google item is trashed by this plan.
    asset_ids: list[str] = field(default_factory=list)
    blocked: bool = False
    reason: str | None = None
    # Rows marked deleted locally *without* trashing anything, because their
    # Google item is still the backup of another live Immich asset.
    retained_ids: list[str] = field(default_factory=list)
    # Rows left untouched because nothing proves which Google item their
    # hash names (legacy rows with no recorded upload hash whose checksum
    # Google no longer maps to the recorded item). Fail closed: they need
    # checking by hand; nothing is trashed for them.
    unverified_ids: list[str] = field(default_factory=list)


class DeletionSweeper:
    def __init__(self, gphotos: GooglePhotosClient, assets: AssetRepo, settings: Settings) -> None:
        self._gphotos = gphotos
        self._assets = assets
        self._settings = settings

    def _verified_legacy_hash(self, candidate: StoredAsset) -> str | None:
        """The hash to trash a pre-hashing row by, only with proof it names *its* item.

        Rows synced before the resolver recorded `upload_sha1` only have
        Immich's `checksum`, which for External Library assets is a virtual
        sha1(path) that names no bytes at all. Rather than assume such rows
        never synced, ask Google (read-only) what that checksum maps to: only
        when it is exactly the media key recorded for this row is the
        checksum used. Anything else fails closed.
        """
        try:
            remote = self._gphotos.exists(candidate.asset.checksum)
        except Exception:  # noqa: BLE001 - no proof, no deletion
            return None
        if remote and remote == candidate.media_key:
            return candidate.asset.checksum
        return None

    def plan(self) -> DeletionPlan:
        if not self._settings.deletions_enabled:
            return DeletionPlan()
        candidates = self._assets.trashed_synced()
        if not candidates:
            return DeletionPlan()
        ok, reason = deletion_allowed(
            count=len(candidates),
            synced_total=self._assets.synced_count(),
            policy=self._settings.deletion_policy,
        )
        if not ok:
            return DeletionPlan(
                asset_ids=[c.asset.immich_id for c in candidates], blocked=True, reason=reason
            )
        trashing = {c.asset.immich_id for c in candidates}
        checksums: list[str] = []
        asset_ids: list[str] = []
        retained: list[str] = []
        unverified: list[str] = []
        planned_keys: set[str] = set()
        for c in candidates:
            assert c.media_key is not None  # trashed_synced only returns rows with one
            # Never trash an item another live Immich asset still maps to.
            others = [
                r
                for r in self._assets.rows_for_media_key(c.media_key)
                if r.asset.immich_id not in trashing
                and r.state is AssetState.SYNCED
                and not r.asset.is_trashed
            ]
            if others:
                retained.append(c.asset.immich_id)
                continue
            # Google trashes by the dedup key derived from the content SHA-1
            # it was given at upload: `upload_sha1` is exactly that for rows
            # synced while the resolver hashed bytes.
            content_hash = c.upload_sha1 or self._verified_legacy_hash(c)
            if content_hash is None:
                unverified.append(c.asset.immich_id)
                continue
            if c.media_key not in planned_keys:
                checksums.append(content_hash)
                planned_keys.add(c.media_key)
            asset_ids.append(c.asset.immich_id)
            # A Motion Photo built for this still *in addition to* its own
            # item (a derived_motion reprocess) is the same photo: it goes
            # with it, by its own recorded hash -- never inferred from the
            # still's.
            live = self._assets.live_photo(c.asset.immich_id)
            if (
                live is not None
                and live["derived_sha1"]
                and live["derived_media_key"]
                and live["derived_media_key"] != c.media_key
                and live["derived_media_key"] not in planned_keys
            ):
                checksums.append(live["derived_sha1"])
                planned_keys.add(live["derived_media_key"])
        return DeletionPlan(
            checksums=checksums,
            asset_ids=asset_ids,
            blocked=False,
            reason=reason,
            retained_ids=retained,
            unverified_ids=unverified,
        )

    def execute(self, plan: DeletionPlan) -> int:
        if plan.blocked:
            return 0
        if plan.checksums:
            self._gphotos.trash(plan.checksums)
            self._assets.mark_deleted(plan.asset_ids)
        if plan.retained_ids:
            self._assets.mark_deleted(plan.retained_ids)
        return len(plan.checksums)
