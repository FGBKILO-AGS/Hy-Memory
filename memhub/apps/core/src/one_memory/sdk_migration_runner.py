"""Offline startup cutover from the legacy Governance mirror to SDK authority.

Every released build wrote memory and raw turns through the SDK first; the
Evidence/Governance tables only mirrored them. The cutover therefore copies no
memory. It makes the surviving write queue independent of the legacy tables,
commits SDK authority and drops those tables.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from one_memory.database_access import main_database
from one_memory.imports.models import EvidenceRecord
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_server.contracts import (
    BatchIdentity,
    WriteRequest,
    internal_session_id_for,
)
from one_memory.memory_server.durable_queue import DurableWriteQueue
from one_memory.memory_server.unified_entry import UnifiedWriteIntake
from one_memory.sdk_data_plane_migration import (
    CutoverMarker,
    DataPlaneMigrationError,
    SdkDataPlaneMigrationStore,
)
from one_memory.sdk_migration_bundle import inspect_sdk_migration_bundle
from one_memory.target_schema import drop_legacy_tables

PREPARED_PHASE = "prepared"


@dataclass(frozen=True)
class CutoverPreparation:
    """What the cutover did to legacy work, for the receipt and the dialog."""

    frozen_jobs: int
    moved_tasks: int
    moved_dispatches: int
    abandoned: dict[str, int]

    def as_document(self) -> dict[str, object]:
        return {
            "frozen_jobs": self.frozen_jobs,
            "moved_tasks": self.moved_tasks,
            "moved_dispatches": self.moved_dispatches,
            "abandoned": dict(self.abandoned),
        }


def _tables(database: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def recover_abandoned_staging(*, data_dir: Path, database_path: Path, namespace: str) -> None:
    """Undo or discard the SDK staging copy that builds through 19 created.

    Those builds staged a full SDK copy and swapped directories at cutover. The
    live root was untouched until the swap, so before SDK authority the
    protective root is the original store; after it, the protective root is
    stale.
    """
    home = data_dir.resolve()
    live_root = home / "hy-memory" / namespace
    authority = SdkDataPlaneMigrationStore(database_path).authority()
    protective_roots = sorted(live_root.parent.glob(f".{namespace}.pre-migration-*"))
    if authority.value != "sdk" and len(protective_roots) > 1:
        raise DataPlaneMigrationError("sdk_migration_swap_state_ambiguous")
    for protective_root in protective_roots:
        if authority.value == "sdk":
            if not live_root.is_dir():
                raise DataPlaneMigrationError("sdk_migration_committed_store_missing")
            shutil.rmtree(protective_root)
            continue
        if live_root.exists():
            shutil.rmtree(live_root)
        protective_root.replace(live_root)
    shutil.rmtree(home / ".sdk-migration", ignore_errors=True)


def abandoned_staging_present(*, data_dir: Path, namespace: str) -> bool:
    home = data_dir.resolve()
    return (home / ".sdk-migration").exists() or any(
        (home / "hy-memory").glob(f".{namespace}.pre-migration-*")
    )


def prepare_legacy_write_queue_for_cutover(database_path: Path) -> tuple[int, int]:
    """Freeze old jobs and move unstarted task rows into the surviving queue."""
    with main_database(database_path).read(row_factory=sqlite3.Row) as database:
        tables = _tables(database)
        if "evidence" not in tables:
            return (0, 0)
        jobs = (
            tuple(database.execute("SELECT job_id,payload_json FROM memory_write_jobs"))
            if "memory_write_jobs" in tables
            else ()
        )
    evidence = EvidenceStore(database_path)
    frozen = 0
    for row in jobs:
        request = WriteRequest.model_validate_json(str(row["payload_json"]))
        if request.input_records:
            continue
        records = tuple(evidence.get(identifier) for identifier in request.evidence_ids)
        if any(record is None for record in records):
            raise DataPlaneMigrationError(f"sdk_data_plane_write_job_input_missing:{row['job_id']}")
        materialized = tuple(record for record in records if record is not None)
        updated = request.model_copy(update={"input_records": materialized})
        with main_database(database_path).write(autocommit=True) as database:
            database.execute(
                "UPDATE memory_write_jobs SET payload_json=? WHERE job_id=?",
                (updated.model_dump_json(), str(row["job_id"])),
            )
        frozen += 1

    if not {"memory_tasks", "memory_task_inputs"} <= tables:
        return (frozen, 0)
    with main_database(database_path).read(row_factory=sqlite3.Row) as database:
        tasks = tuple(
            database.execute(
                "SELECT task.task_id,task.input_ref,task.status,"
                "task.hy_invocation_count,inputs.refs_json "
                "FROM memory_tasks task "
                "LEFT JOIN memory_task_inputs inputs USING(task_id) "
                "WHERE task.status IN "
                "('pending','waiting_for_permission','waiting_for_model',"
                "'deferred_outside_window','running')"
            )
        )
    queue = DurableWriteQueue(database_path)
    moved = 0
    for task in tasks:
        if int(task["hy_invocation_count"] or 0) > 0:
            continue
        references = json.loads(str(task["refs_json"])) if task["refs_json"] else []
        identifiers = [
            str(reference["evidence_id"])
            for reference in references
            if isinstance(reference, dict) and reference.get("evidence_id")
        ] or [str(task["input_ref"])]
        records = tuple(evidence.get(identifier) for identifier in identifiers)
        if any(record is None for record in records):
            raise DataPlaneMigrationError(
                f"sdk_data_plane_memory_task_input_missing:{task['task_id']}"
            )
        materialized = tuple(record for record in records if record is not None)
        source = materialized[0]
        internal_session_id = internal_session_id_for(
            source.conversation_id,
            source.source_id,
        )
        queue.submit(
            WriteRequest(
                internal_session_id=internal_session_id,
                source_id=source.source_id,
                batch_identity=BatchIdentity(
                    internal_session_id=internal_session_id,
                    first_round=1,
                    last_round=max(1, len(materialized)),
                ),
                evidence_ids=[record.evidence_id for record in materialized],
                input_records=materialized,
                idempotency_key=f"legacy-task:{task['task_id']}",
                priority="normal",
            )
        )
        with main_database(database_path).write(autocommit=True) as database:
            database.execute(
                "UPDATE memory_tasks SET status='cancelled',runner_token=NULL,"
                "last_error_code='migrated_to_memory_write_job',updated_at=CURRENT_TIMESTAMP "
                "WHERE task_id=?",
                (str(task["task_id"]),),
            )
        moved += 1
    return (frozen, moved)


def move_pending_dispatches(database_path: Path) -> int:
    """Queue Evidence that was waiting for its source file to finish.

    Its source cursor has already moved past these rows, so History catch-up
    after the cutover would not read them again.
    """
    with main_database(database_path).read(row_factory=sqlite3.Row) as database:
        if not {"evidence", "evidence_pending_memory_dispatch"} <= _tables(database):
            return 0
        identifiers = tuple(
            str(row["evidence_id"])
            for row in database.execute(
                "SELECT DISTINCT pending.evidence_id "
                "FROM evidence_pending_memory_dispatch pending "
                "JOIN evidence USING(evidence_id) "
                "ORDER BY COALESCE(evidence.created_at,evidence.imported_at),"
                "COALESCE(evidence.ingest_seq,evidence.rowid),pending.evidence_id"
            )
        )
    evidence = EvidenceStore(database_path)
    conversations: dict[tuple[str, str, str, str | None], list[EvidenceRecord]] = {}
    for identifier in identifiers:
        record = evidence.get(identifier)
        if record is None:
            continue
        key = (record.user_id, record.source_id, record.conversation_id, record.workspace)
        conversations.setdefault(key, []).append(record)
    intake = UnifiedWriteIntake(DurableWriteQueue(database_path))
    for records in conversations.values():
        intake.submit_records(tuple(records))
    return len(conversations)


def abandoned_legacy_work(database_path: Path) -> dict[str, int]:
    """Count legacy work that ends with the cutover instead of blocking it.

    A task that may have reached the SDK is not replayed: the SDK already holds
    whatever it committed. Pending user deletions are not carried over.
    """
    abandoned: dict[str, int] = {}
    with main_database(database_path).read(row_factory=sqlite3.Row) as database:
        tables = _tables(database)
        checks = (
            (
                "invoked_tasks",
                "memory_tasks",
                "status='unknown' OR (status='running' AND hy_invocation_count > 0)",
            ),
            ("deletions", "memory_deletions", "status IN ('pending','running')"),
        )
        for name, table, where in checks:
            if table not in tables:
                continue
            count = int(
                database.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]
            )
            if count:
                abandoned[name] = count
    return abandoned


def prepare_legacy_work_for_cutover(database_path: Path) -> CutoverPreparation:
    """Make the surviving queue self-contained. Safe to repeat."""
    frozen, moved = prepare_legacy_write_queue_for_cutover(database_path)
    dispatches = move_pending_dispatches(database_path)
    return CutoverPreparation(
        frozen_jobs=frozen,
        moved_tasks=moved,
        moved_dispatches=dispatches,
        abandoned=abandoned_legacy_work(database_path),
    )


def commit_sdk_cutover(
    *,
    data_dir: Path,
    database_path: Path,
    migration_id: str,
    bundle_id: str,
    sdk_version: str,
    store_version: str = "1",
) -> CutoverMarker:
    """Verify the backup, settle legacy work and commit SDK authority."""
    inspect_sdk_migration_bundle(data_dir=data_dir, bundle_id=bundle_id)
    # Repeated here because a CLI-managed Runtime may have run on legacy
    # authority since `sdk-migrate`.
    preparation = prepare_legacy_work_for_cutover(database_path)
    digest = hashlib.sha256(
        json.dumps(
            preparation.as_document(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return SdkDataPlaneMigrationStore(database_path).commit_cutover(
        migration_id=migration_id,
        sdk_version=sdk_version,
        store_version=store_version,
        backup_bundle_id=bundle_id,
        reconciliation_digest=digest,
    )


def resume_legacy_cleanup(*, data_dir: Path, database_path: Path) -> tuple[str, ...]:
    """Finish a cleanup interrupted after the durable authority switch."""
    store = SdkDataPlaneMigrationStore(database_path)
    if store.authority().value != "sdk":
        raise DataPlaneMigrationError("sdk_data_plane_cleanup_before_cutover")
    marker = store.cutover()
    if marker is not None:
        inspect_sdk_migration_bundle(
            data_dir=data_dir,
            bundle_id=marker.backup_bundle_id,
        )
    from one_memory.hy_runtime_config import HY_MEMORY_NAMESPACE

    recover_abandoned_staging(
        data_dir=data_dir,
        database_path=database_path,
        namespace=HY_MEMORY_NAMESPACE,
    )
    return drop_legacy_tables(database_path)
