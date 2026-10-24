# SDK Data Plane Migration v1

Updated: 2026-10-24.

This contract moves released MemHub installations from the historical
Evidence/Governance authority to the Hy Memory SDK data plane defined by
[`../ARCHITECTURE.md`](../ARCHITECTURE.md).

It is an offline migration executed inside the App's visible startup loading
phase, before Core accepts product traffic. It is not a dual-read or dual-write
design and not a separate manual upgrade flow for ordinary users.

## 1. Why no memory is copied

Every released build (`v1.0.0` through build 19) wrote memory through SDK
`add()` / `add_extracted()` first. The SDK stored the raw turn (L1 raw) and the
extracted Chat / Coding Experience / Working Fact memories in its own stores.
The Governance and Evidence tables were a mirror that MemHub read back from the
SDK afterwards.

The SDK stores are therefore already complete. The migration copies no memory
and no raw turn, and does not compare the mirror with the SDK:

- the mirror's layer labels did not record which Coding store held a row, and
  the SDK physically deletes experiences its reconciler retires, so the mirror
  cannot be trusted to decide what the SDK should contain;
- the product has no user editing of memories in practice, so the mirror holds
  no user intent the SDK lacks;
- SDK L1 raw is retained for three days by the Runtime's own
  `prune_expired_raw`, not by the migration.

## 2. Authority before and after cutover

Before the cutover marker is committed, legacy tables remain the product read
authority and Core may still run on them (CLI-managed Runtime). The App does
not start Runtime while migration is incomplete.

After the cutover marker is committed:

- SDK stores are the only memory data authority;
- MemHub stores management state and `memory_id` / raw references only;
- legacy Evidence/Governance/retrieval/session-shadow and Matter tables are
  dropped before the normal App window is revealed;
- startup never falls back to legacy authority.

The surviving `memory_authority_state` singleton, not table existence, selects
the authority.

A fresh home bootstraps the latest management schema with SDK authority
directly and never creates legacy tables.

## 3. Startup steps

The packaged App runs these Core commands while the startup placeholder is
visible, under the Runtime maintenance fence and the capture fence (host
capture hooks return silently instead of writing Evidence):

1. `migration-bundle`: copy `one-memory.sqlite3`, the SDK data plane and the
   non-secret configuration into a versioned bundle, then verify it by an
   isolated restore. An existing verified bundle for the same migration is
   reused.
2. `sdk-migrate`: apply the management schema migrations, discard SDK staging
   left by builds through 19 (see §5), and make legacy queue work
   self-contained (see §4). It needs no model profile and no SDK client. The run
   is recorded as `phase=prepared, state=ready`.
3. `sdk-cutover`: re-verify the bundle, repeat §4 (a CLI-managed Runtime may
   have run since step 2), commit SDK authority, and drop the legacy tables.

Each step is safe to repeat. A retry resumes the same `sdk-authority-v2-*`
migration ID and bundle, including a run that build 19 left in
`review_required/blocked`; earlier per-item ledger rows are ignored and dropped
with the legacy tables.

After cutover, History catch-up scans forward from each source checkpoint and
recovers turns skipped while capture was fenced.

## 4. Queue work

Queue rows remain MemHub management state. Before the legacy tables disappear:

- each `memory_write_jobs` row that only names Evidence IDs gets those records
  frozen into `payload_json`;
- unstarted legacy `memory_tasks` become
  `memory_write_jobs(idempotency_key="legacy-task:<task_id>")`;
- Evidence waiting in `evidence_pending_memory_dispatch` (its source cursor has
  already passed it) becomes one write job per conversation.

Legacy work that cannot continue does not block cutover; its counts are
reported in the `sdk-migrate` / cutover receipt as `abandoned`:

- a legacy task that may already have reached the SDK (`unknown`, or `running`
  with an SDK invocation) is not replayed: the SDK already holds whatever it
  committed;
- pending legacy user deletions are not carried over.

Surviving `memory_write_jobs` with `hy_invoked=1` keep their normal
post-cutover handling through SDK `get_write_outcome(request_id)`.

## 5. Staging left by earlier builds

Builds through 19 copied the SDK root to `<home>/.sdk-migration/` and swapped
directories at cutover. Before SDK authority, a leftover
`hy-memory/.<namespace>.pre-migration-*` directory is the original store and is
restored; after SDK authority it is stale and removed. More than one such
directory before authority is ambiguous and fails closed
(`sdk_migration_swap_state_ambiguous`).

## 6. Failure and rollback

Any failure leaves the marker absent and reports a stable code to the startup
dialog (for example `sdk_migration_disk_space_insufficient`,
`sdk_migration_bundle_copy_failed`, `sdk_migration_preflight_failed`,
`sdk_migration_cutover_failed`). The next startup retries.

After cutover but before any SDK-only write, rollback restores the bundle
together with the matching old Runtime. After an SDK-only write, automatic
binary-only rollback is forbidden; restoring the pre-cutover bundle loses the
post-cutover writes and must be explicit. An old Runtime must fail closed on a
post-cutover home. Changing only the Runtime `current` symlink is not data
rollback.

## 7. Acceptance

The release gate covers:

- a fresh install with no legacy table names;
- a legacy home reaching cutover with its SDK stores byte-for-byte unchanged;
- a home left blocked by build 19 (quarantined ledger and staging copy)
  completing on the next startup;
- pending job freezing, legacy task and pending dispatch conversion;
- invoked/unknown legacy tasks and pending deletions not blocking cutover;
- interrupted directory swap recovery before and after authority;
- bundle creation, reuse and restore;
- physical absence of every legacy table after cutover.
