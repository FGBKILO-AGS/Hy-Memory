"""Offline maintenance of the HY SDK data plane for App migrations.

Runtime product code must use public ``HyMemoryClient`` methods.  The startup
migration boundary is the isolated exception: all writers are stopped, so it
may copy, restore and repair SDK-owned local stores.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hy_memory import (  # type: ignore[import-untyped]
    DataPlaneRestore,
    DataPlaneSnapshot,
    restore_local_data_plane,
    snapshot_local_data_plane,
)

from one_memory.database_access import main_database
from one_memory.hy_runtime_config import HY_MEMORY_NAMESPACE

_METADATA_REPAIR_VERSION = 1
_BATCH = 500


@dataclass(frozen=True)
class RepairMetadata:
    source: str | None
    memory_at: int | None
    created_at: int | None


@dataclass(frozen=True)
class MetadataRepairResult:
    migration_id: str | None
    mapped: int
    vector_rows: int
    coding_rows: int


def snapshot_data_plane(
    *,
    source_root: Path,
    target_root: Path,
    writers_stopped: bool,
) -> DataPlaneSnapshot:
    """Snapshot an explicitly quiesced SDK-owned local data plane."""
    return snapshot_local_data_plane(
        source_root=source_root,
        target_root=target_root,
        writers_stopped=writers_stopped,
    )


def restore_data_plane(
    *,
    snapshot_root: Path,
    target_root: Path,
    writers_stopped: bool,
    protective_root: Path | None = None,
) -> DataPlaneRestore:
    """Restore an explicitly quiesced SDK-owned local data plane."""
    return restore_local_data_plane(
        snapshot_root=snapshot_root,
        target_root=target_root,
        writers_stopped=writers_stopped,
        protective_root=protective_root,
    )


def repair_sdk_metadata_from_bundle(
    *,
    data_dir: Path,
    database_path: Path,
    now: datetime | None = None,
) -> MetadataRepairResult:
    """Restore metadata lost by historical cutovers from their verified bundle."""
    migration_id = _sdk_migration_id(database_path)
    if migration_id is None:
        return MetadataRepairResult(None, 0, 0, 0)
    receipt = data_dir / f".sdk-metadata-repair-v{_METADATA_REPAIR_VERSION}-{migration_id}.json"
    if receipt.is_file():
        document = json.loads(receipt.read_text(encoding="utf-8"))
        return MetadataRepairResult(
            migration_id,
            int(document.get("mapped", 0)),
            int(document.get("vector_rows", 0)),
            int(document.get("coding_rows", 0)),
        )
    legacy = (
        data_dir
        / "backups"
        / f"{migration_id}-pre-cutover.sdk-bundle"
        / "management"
        / "one-memory.sqlite3"
    )
    if not legacy.is_file():
        return MetadataRepairResult(migration_id, 0, 0, 0)
    mapping = _legacy_metadata(legacy, now=now or datetime.now(UTC))
    vector_rows = _repair_vector(data_dir, mapping)
    coding_rows = _repair_coding(data_dir, mapping)
    result = MetadataRepairResult(
        migration_id,
        len(mapping),
        vector_rows,
        coding_rows,
    )
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "migration_id": migration_id,
                "mapped": result.mapped,
                "vector_rows": vector_rows,
                "coding_rows": coding_rows,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt)
    return result


def _sdk_migration_id(database_path: Path) -> str | None:
    with main_database(database_path).read(row_factory=sqlite3.Row) as database:
        row = database.execute(
            "SELECT authority,migration_id FROM memory_authority_state WHERE singleton=1"
        ).fetchone()
    if row is None or row["authority"] != "sdk":
        return None
    value = row["migration_id"]
    return str(value) if isinstance(value, str) and value else None


def _legacy_metadata(path: Path, *, now: datetime) -> dict[str, RepairMetadata]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as database:
        database.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {"memory_task_outputs", "governed_memories", "evidence"} <= tables:
            return {}
        outputs: list[tuple[str, str, tuple[str, ...]]] = []
        evidence_ids: set[str] = set()
        for row in database.execute("SELECT body FROM memory_task_outputs"):
            try:
                body = json.loads(str(row["body"]))
            except (TypeError, ValueError):
                continue
            legacy_id = body.get("memory_id")
            canonical_id = body.get("hy_memory_id")
            refs = body.get("support_refs")
            if not isinstance(legacy_id, str) or not isinstance(canonical_id, str):
                continue
            identifiers = (
                tuple(
                    str(ref["evidence_id"])
                    for ref in refs
                    if isinstance(ref, dict) and isinstance(ref.get("evidence_id"), str)
                )
                if isinstance(refs, list)
                else ()
            )
            outputs.append((legacy_id, canonical_id, identifiers))
            evidence_ids.update(identifiers)
        evidence: dict[str, tuple[str, datetime | None]] = {}
        for batch in _batches(sorted(evidence_ids)):
            placeholders = ",".join("?" for _ in batch)
            for row in database.execute(
                "SELECT evidence_id,source_id,created_at,imported_at "
                f"FROM evidence WHERE evidence_id IN ({placeholders})",
                batch,
            ):
                evidence[str(row["evidence_id"])] = (
                    str(row["source_id"]),
                    _plausible_time(
                        row["created_at"],
                        fallback=row["imported_at"],
                        now=now,
                    ),
                )
        governed = {
            str(row["memory_id"]): _plausible_time(
                row["created_at"],
                fallback=row["updated_at"],
                now=now,
            )
            for row in database.execute(
                "SELECT memory_id,created_at,updated_at FROM governed_memories"
            )
        }
    # Several legacy memories can fold into one SDK memory. Pool their evidence
    # before deciding, so a disagreeing source cannot be overwritten by whichever
    # output happened to come last.
    pooled: dict[str, tuple[set[str], list[datetime], list[datetime]]] = {}
    for legacy_id, canonical_id, refs in outputs:
        sources, times, created = pooled.setdefault(canonical_id, (set(), [], []))
        for identifier in refs:
            if identifier not in evidence:
                continue
            source, timestamp = evidence[identifier]
            if source and source != "default":
                sources.add(source)
            if timestamp is not None:
                times.append(timestamp)
        legacy_created = governed.get(legacy_id)
        if legacy_created is not None:
            created.append(legacy_created)
    return {
        canonical_id: RepairMetadata(
            source=next(iter(sources)) if len(sources) == 1 else None,
            memory_at=int(max(times).timestamp()) if times else None,
            created_at=int(min(created).timestamp()) if created else None,
        )
        for canonical_id, (sources, times, created) in pooled.items()
    }


def _plausible_time(
    value: object,
    *,
    fallback: object,
    now: datetime,
) -> datetime | None:
    for candidate in (value, fallback):
        if not isinstance(candidate, str) or not candidate:
            continue
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        parsed = parsed.astimezone(UTC)
        if parsed <= now + timedelta(days=1):
            return parsed
    return None


def _repair_vector(
    data_dir: Path,
    mapping: dict[str, RepairMetadata],
) -> int:
    directory = data_dir / "hy-memory" / HY_MEMORY_NAMESPACE / "vector"
    if not mapping or not (directory / "chroma.sqlite3").is_file():
        return 0
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(directory),
        settings=Settings(anonymized_telemetry=False),
    )
    # The SDK client is not open yet (startup has not created its handle), so
    # this is the only writer; release the store before the SDK takes it.
    try:
        repaired = 0
        ids = tuple(mapping)
        for collection in client.list_collections():
            for batch in _batches(ids):
                page = collection.get(ids=list(batch), include=["metadatas"])
                updated_ids: list[str] = []
                updated_metadata: list[dict[str, object]] = []
                for memory_id, metadata in zip(
                    page.get("ids") or (),
                    page.get("metadatas") or (),
                    strict=True,
                ):
                    if not isinstance(metadata, dict):
                        continue
                    desired = mapping.get(str(memory_id))
                    if desired is None:
                        continue
                    next_metadata = dict(metadata)
                    changed = False
                    if desired.source and next_metadata.get("source") in {
                        None,
                        "",
                        "default",
                    }:
                        next_metadata["source"] = desired.source
                        changed = True
                    if (
                        desired.memory_at is not None
                        and next_metadata.get("memory_at") != desired.memory_at
                    ):
                        next_metadata["memory_at"] = desired.memory_at
                        changed = True
                    if (
                        desired.created_at is not None
                        and next_metadata.get("gmt_created") != desired.created_at
                    ):
                        next_metadata["gmt_created"] = desired.created_at
                        changed = True
                    if changed:
                        updated_ids.append(str(memory_id))
                        updated_metadata.append(next_metadata)
                if updated_ids:
                    collection.update(ids=updated_ids, metadatas=updated_metadata)
                    repaired += len(updated_ids)
    finally:
        client.close()
    return repaired


def _repair_coding(
    data_dir: Path,
    mapping: dict[str, RepairMetadata],
) -> int:
    root = data_dir / "hy-memory" / HY_MEMORY_NAMESPACE
    repaired = 0
    for name in ("coding.sqlite3", "coding_cv.sqlite3"):
        path = root / name
        if not path.is_file():
            continue
        with closing(sqlite3.connect(path)) as database:
            tables = {
                str(row[0])
                for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            for table, identifier in (
                ("coding_memory_meta", "experience_id"),
                ("working_fact", "fact_id"),
            ):
                if table not in tables:
                    continue
                rows = database.execute(
                    f"SELECT {identifier},source,created_at FROM {table}"
                ).fetchall()
                for memory_id, source, created_at in rows:
                    desired = mapping.get(str(memory_id))
                    if desired is None:
                        continue
                    updates: list[str] = []
                    values: list[object] = []
                    if desired.source and source in {None, "", "default"}:
                        updates.append("source=?")
                        values.append(desired.source)
                    if desired.created_at is not None:
                        desired_created_at = (
                            datetime.fromtimestamp(desired.created_at, tz=UTC)
                            .replace(tzinfo=None)
                            .isoformat()
                        )
                        if created_at != desired_created_at:
                            updates.append("created_at=?")
                            values.append(desired_created_at)
                    if not updates:
                        continue
                    values.append(memory_id)
                    database.execute(
                        f"UPDATE {table} SET {','.join(updates)} WHERE {identifier}=?",
                        values,
                    )
                    repaired += 1
            database.commit()
    return repaired


def _batches[T](values: list[T] | tuple[T, ...]) -> Iterator[tuple[T, ...]]:
    for start in range(0, len(values), _BATCH):
        yield tuple(values[start : start + _BATCH])
