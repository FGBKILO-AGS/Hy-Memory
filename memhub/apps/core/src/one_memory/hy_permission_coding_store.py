"""Operation-bound read/write capability for the pinned native HY coding store.

HY's coding route writes two layers, experiences and working facts. Rows this
lease cannot attest stay in the same backend but are invisible here, so they
never reach a coding prompt and cannot be updated, superseded or linked.

HY's reconcilers catch store exceptions and carry on, so a denial latches and
recheck() raises it again at finish; a swallowed denial still fails the task.
Any native method not implemented here is denied rather than passed through.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from types import MappingProxyType
from typing import Any, NoReturn

from hy_memory.coding.types import CodingMemory, WorkingFact  # type: ignore[import-untyped]

from one_memory.hy_permission_store import HyNodePermission
from one_memory.model_processing_permission_guard import (
    ModelProcessingPermissionLease,
    permission_error,
)

_NATIVE_WRITES = frozenset(
    {
        "insert",
        "update",
        "delete",
        "insert_fact",
        "supersede_fact",
        "set_fact_derived_experiences",
        "set_experience_source_facts",
        # Trace rows are raw conversation: writes must be gated like any other.
        "append_trace",
    }
)

_TIMESTAMPS = frozenset({"created_at", "updated_at"})

# SDK 2.0.2 reads `config` off the coding store while routing a coding
# turn. The wrapper must answer it, but the native `MemoryConfig` carries
# credentials (llm/embedder api keys, graph and cache passwords) and is a live
# handle into the backend, so it is never handed out. Only this redacted
# projection is exposed: credential-bearing keys collapse to None.
_CONFIG_REDACTED_KEYS = frozenset(
    {
        "api_key",
        "password",
        "username",
        "mysql_password",
        "mysql_user",
        "extra_headers",
        "extra_body",
    }
)


class _RedactedConfig(Mapping[str, Any]):
    """Read-only config view: readable by key and attribute, never the native one.

    The SDK reaches config values both ways, so this keeps attribute access
    working. It is deliberately not the native MemoryConfig: it exposes only
    plain copied values, so no backend handle, connection or callable escapes.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        try:
            return values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("redacted config is read-only")

    def __repr__(self) -> str:
        return f"_RedactedConfig({sorted(self._values)})"


def _redact_config(value: Any, depth: int = 0) -> Any:
    """Copy config data into read-only views, collapsing credentials to None.

    The SDK reads this both by attribute and by key, so each config section
    becomes a mapping-like view rather than a plain dict: attribute access stays
    available while the native object itself is never handed out.
    """
    if depth > 4:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _RedactedConfig(
            {
                key: None if key in _CONFIG_REDACTED_KEYS else _redact_config(item, depth + 1)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return [_redact_config(item, depth + 1) for item in value]
    if hasattr(value, "__dict__") and type(value).__module__.startswith("hy_memory"):
        return _redact_config(vars(value), depth + 1)
    return None


def _config_projection(native: Any) -> Any:
    try:
        return _redact_config(native.config)
    except Exception:
        return None


def hy_coding_fingerprint(row: CodingMemory | WorkingFact) -> str:
    """Hash the complete native row as read back, including links and timestamps."""
    if type(row) is CodingMemory:
        kind = "experience"
    elif type(row) is WorkingFact:
        kind = "fact"
    else:
        raise permission_error("hy_coding_store_row_shape")
    payload = {"kind": kind, **row.to_dict()}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _row_id(row: CodingMemory | WorkingFact) -> str:
    return str(row.experience_id if type(row) is CodingMemory else row.fact_id)


class HyPermissionCodingStore:
    """Closed, task-exclusive adapter; initialize/close never own the native store.

    node_permissions is the same attested ledger the chat vector store uses;
    coding rows are keyed by experience or fact id. written_nodes carries the
    whole lease footprint for each row written here, as the chat store does.
    """

    def __init__(
        self,
        native: Any,
        lease: ModelProcessingPermissionLease,
        *,
        agent_id: str,
        session_id: str,
        node_permissions: Mapping[str, HyNodePermission] | None = None,
    ) -> None:
        self._native = native
        self._lease = lease
        self._agent_id = agent_id
        self._session_id = session_id
        self._user_id = lease.user_id
        self._failure: str | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._deleted: set[str] = set()
        self._written: dict[str, HyNodePermission] = {}
        self._superseded: dict[str, str] = {}
        self._evidence_ids = tuple(record.evidence_id for record in lease.records)
        self._permissions = {
            key: proof
            for key, proof in (node_permissions or {}).items()
            if set(proof.evidence_ids) <= set(self._evidence_ids)
        }
        self.recheck()
        if not agent_id or agent_id == "*" or not session_id or not self._evidence_ids:
            self._deny("scope")

    def __getattr__(self, name: str) -> NoReturn:
        if name.startswith("__"):
            raise AttributeError(name)
        self._deny("unsupported_operation")

    @property
    def written_nodes(self) -> Mapping[str, HyNodePermission]:
        self.recheck()
        return MappingProxyType(dict(self._written))

    @property
    def config(self) -> Any:
        """Read-only, redacted view of the native coding config.

        SDK 2.0.2 reads `config` when it routes a coding turn. Answering
        it is safe only as a projection: the native MemoryConfig is a live
        backend handle and carries credentials, so callers get plain values
        with every credential-bearing key collapsed to None. This carries no
        identity, permission or tenant state and is not the native object.
        """
        self._assert_open()
        return _config_projection(self._native)

    @property
    def retired_ids(self) -> frozenset[str]:
        """Experiences HY deleted in this operation."""
        self.recheck()
        return frozenset(self._deleted)

    @property
    def superseded_ids(self) -> Mapping[str, tuple[str, ...]]:
        """Facts HY superseded in this operation, with the fact that replaced each.

        A successor linked before it existed must have been written here by
        now; otherwise the old fact points at a row nobody attested.
        """
        self.recheck()
        if any(new not in self._written for new in self._superseded.values()):
            self._deny("link_denied")
        return MappingProxyType({old: (new,) for old, new in self._superseded.items()})

    def recheck(self) -> None:
        self._assert_open()
        try:
            self._lease.recheck()
        except Exception as error:
            code = str(error)
            self._failure = (
                code
                if code.startswith("model_content_") and len(code) <= 100
                else "model_content_hy_coding_store_lease_failed"
            )
            raise permission_error(self._failure) from None

    def _assert_open(self) -> None:
        if self._closed:
            self._deny("closed")
        if self._failure:
            raise permission_error(self._failure)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._permissions.clear()
            self._written.clear()

    def _deny(self, suffix: str) -> NoReturn:
        self._failure = self._failure or f"model_content_hy_coding_store_{suffix}"
        raise permission_error(self._failure)

    def _scope(self, user_id: str | None) -> None:
        if user_id not in (None, self._user_id):
            self._deny("scope")

    async def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        writing = method in _NATIVE_WRITES
        self.recheck() if writing else self._assert_open()
        try:
            result = await getattr(self._native, method)(*args, **kwargs)
        except Exception:
            self._deny("backend_failed")
        self.recheck() if writing else self._assert_open()
        return result

    def _visible(self, row: Any, expected: Any) -> bool:
        if row is None:
            return False
        if type(row) is not expected:
            self._deny("row_shape")
        key = _row_id(row)
        proof = self._permissions.get(key)
        if proof is None or key in self._deleted or row.user_id != self._user_id:
            return False
        if hy_coding_fingerprint(row) != proof.fingerprint:
            self._deny("row_changed")
        return True

    async def _experience(self, key: str) -> CodingMemory | None:
        if key not in self._permissions or key in self._deleted:
            return None
        row = await self._invoke("get_by_id", key, user_id=self._user_id)
        return deepcopy(row) if self._visible(row, CodingMemory) else None

    async def _fact(self, key: str) -> WorkingFact | None:
        if key not in self._permissions or key in self._deleted:
            return None
        rows = await self._invoke("get_facts", [key], user_id=self._user_id)
        row = rows[0] if rows else None
        return deepcopy(row) if self._visible(row, WorkingFact) else None

    def _new_row_in_scope(self, row: CodingMemory | WorkingFact) -> bool:
        return bool(
            row.user_id == self._user_id
            and row.agent_id == self._agent_id
            and row.session_id == self._session_id
        )

    async def _capture(
        self, key: str, kind: Any, expected: CodingMemory | WorkingFact | None = None
    ) -> None:
        row: Any
        if kind is CodingMemory:
            row = await self._invoke("get_by_id", key, user_id=self._user_id)
        else:
            rows = await self._invoke("get_facts", [key], user_id=self._user_id)
            row = rows[0] if rows else None
        if type(row) is not kind or row.user_id != self._user_id:
            self._deny("write_mismatch")
        if expected is not None:
            stored, planned = row.to_dict(), expected.to_dict()
            for field in _TIMESTAMPS:
                stored.pop(field, None)
                planned.pop(field, None)
            if stored != planned:
                self._deny("write_mismatch")
        proof = HyNodePermission(hy_coding_fingerprint(row), self._evidence_ids)
        self._permissions[key] = proof
        self._written[key] = proof

    async def _plan_new(self, row: CodingMemory | WorkingFact, kind: Any) -> None:
        if type(row) is not kind or not self._new_row_in_scope(row):
            self._deny("scope")
        key = _row_id(row)
        if not key or key in self._permissions or key in self._deleted:
            self._deny("scope")
        if kind is CodingMemory:
            existing = await self._invoke("get_by_id", key)
        else:
            existing = next(iter(await self._invoke("get_facts", [key])), None)
        if existing is not None:
            self._deny("scope")

    # Reads

    async def get_by_id(
        self, experience_id: str, user_id: str | None = None
    ) -> CodingMemory | None:
        async with self._lock:
            self._scope(user_id)
            return await self._experience(experience_id)

    async def get_many(
        self, experience_ids: list[str], user_id: str | None = None
    ) -> list[CodingMemory]:
        async with self._lock:
            self._scope(user_id)
            wanted = [
                key
                for key in dict.fromkeys(experience_ids)
                if key in self._permissions and key not in self._deleted
            ]
            if not wanted:
                return []
            rows = await self._invoke("get_many", wanted, user_id=self._user_id)
            return [deepcopy(row) for row in rows if self._visible(row, CodingMemory)]

    async def search_by_query_embedding(
        self,
        query_embedding: list[float],
        user_id: str,
        *,
        workspace_id: str | None = None,
        branch: str | None = None,
        top: int = 40,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            self._scope(user_id)
            hits = await self._invoke(
                "search_by_query_embedding",
                deepcopy(query_embedding),
                self._user_id,
                workspace_id=workspace_id,
                branch=branch,
                top=top * 4,
            )
            result = []
            for hit in hits:
                key = hit.get("experience_id")
                if not isinstance(key, str) or await self._experience(key) is None:
                    continue
                result.append(
                    {
                        "experience_id": key,
                        "score": float(hit.get("score", 0.0)),
                        "matched_key_kind": hit.get("matched_key_kind"),
                        "matched_key_text": hit.get("matched_key_text"),
                    }
                )
            return result[:top]

    async def get_fact(self, fact_id: str, user_id: str | None = None) -> WorkingFact | None:
        async with self._lock:
            self._scope(user_id)
            return await self._fact(fact_id)

    async def get_facts(self, fact_ids: list[str], user_id: str | None = None) -> list[WorkingFact]:
        async with self._lock:
            self._scope(user_id)
            found = [await self._fact(key) for key in dict.fromkeys(fact_ids)]
            return [row for row in found if row is not None]

    async def search_facts(
        self,
        query_embedding: list[float],
        user_id: str,
        *,
        workspace_id: str | None = None,
        branch: str | None = None,
        top: int = 5,
        query_text: str | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[WorkingFact, float]]:
        async with self._lock:
            self._scope(user_id)
            hits = await self._invoke(
                "search_facts",
                deepcopy(query_embedding),
                self._user_id,
                workspace_id=workspace_id,
                branch=branch,
                top=top * 4,
                query_text=query_text,
                min_score=min_score,
            )
            result = []
            for row, score in hits:
                if self._visible(row, WorkingFact):
                    result.append((deepcopy(row), float(score)))
            return result[:top]

    async def get_fact_ancestors(
        self, fact_id: str, *, user_id: str | None = None, max_depth: int = 8
    ) -> list[WorkingFact]:
        async with self._lock:
            self._scope(user_id)
            if await self._fact(fact_id) is None:
                return []
            rows = await self._invoke(
                "get_fact_ancestors", fact_id, user_id=self._user_id, max_depth=max_depth
            )
            # Oldest first; the chain is only as long as its nearest unapproved link.
            kept: list[WorkingFact] = []
            for row in reversed(rows):
                if not self._visible(row, WorkingFact):
                    break
                kept.append(deepcopy(row))
            return list(reversed(kept))

    # Writes

    async def insert(self, memory: CodingMemory) -> None:
        async with self._lock:
            self.recheck()
            await self._plan_new(memory, CodingMemory)
            await self._invoke("insert", memory)
            await self._capture(memory.experience_id, CodingMemory, memory)

    async def update(self, memory: CodingMemory) -> None:
        async with self._lock:
            self.recheck()
            if type(memory) is not CodingMemory or not self._new_row_in_scope(memory):
                self._deny("scope")
            if await self._experience(memory.experience_id) is None:
                self._deny("target_denied")
            await self._invoke("update", memory)
            await self._capture(memory.experience_id, CodingMemory, memory)

    async def delete(self, experience_id: str, user_id: str) -> bool:
        async with self._lock:
            self._scope(user_id)
            if await self._experience(experience_id) is None:
                self._deny("target_denied")
            if await self._invoke("delete", experience_id, user_id=self._user_id) is not True:
                self._deny("write_failed")
            if await self._invoke("get_by_id", experience_id) is not None:
                self._deny("write_mismatch")
            self._deleted.add(experience_id)
            self._permissions.pop(experience_id, None)
            self._written.pop(experience_id, None)
            return True

    async def insert_fact(self, fact: WorkingFact) -> None:
        async with self._lock:
            self.recheck()
            await self._plan_new(fact, WorkingFact)
            await self._invoke("insert_fact", fact)
            await self._capture(fact.fact_id, WorkingFact, fact)

    async def supersede_fact(self, old_fact_id: str, new_fact_id: str) -> bool:
        async with self._lock:
            self.recheck()
            if await self._fact(old_fact_id) is None or old_fact_id == new_fact_id:
                self._deny("target_denied")
            # HY links the old fact before it inserts the new one. A successor
            # that does not exist yet may only become a row this lease writes;
            # an existing row it cannot attest stays out of reach.
            if await self._fact(new_fact_id) is None:
                if not new_fact_id or new_fact_id in self._deleted:
                    self._deny("target_denied")
                if next(iter(await self._invoke("get_facts", [new_fact_id])), None) is not None:
                    self._deny("target_denied")
            changed = await self._invoke("supersede_fact", old_fact_id, new_fact_id)
            await self._capture(old_fact_id, WorkingFact)
            if changed:
                self._superseded[old_fact_id] = new_fact_id
            return bool(changed)

    async def set_fact_derived_experiences(self, fact_id: str, experience_ids: list[str]) -> None:
        async with self._lock:
            self.recheck()
            if await self._fact(fact_id) is None:
                self._deny("target_denied")
            for key in experience_ids:
                if await self._experience(key) is None:
                    self._deny("link_denied")
            await self._invoke("set_fact_derived_experiences", fact_id, list(experience_ids))
            await self._capture(fact_id, WorkingFact)

    async def set_experience_source_facts(self, experience_id: str, fact_ids: list[str]) -> None:
        async with self._lock:
            self.recheck()
            if await self._experience(experience_id) is None:
                self._deny("target_denied")
            for key in fact_ids:
                if await self._fact(key) is None:
                    self._deny("link_denied")
            await self._invoke("set_experience_source_facts", experience_id, list(fact_ids))
            await self._capture(experience_id, CodingMemory)

    async def get_recent_trace_messages(
        self,
        *,
        user_id: str,
        agent_id: str = "default_agent",
        max_user_turns: int = 5,
    ) -> list[Any]:
        """Read recent raw trace rows, still bounded by this lease.

        Public SDK 2.0.2 asks for recent history before a coding write. The native call is
        scoped to the lease's own user and agent, and rows this lease cannot
        attest are filtered out, so unapproved conversations never reach a
        coding prompt through this path.
        """
        self._scope(user_id)
        if agent_id not in (self._agent_id, "*"):
            self._deny("scope")
        rows = await self._invoke(
            "get_recent_trace_messages",
            user_id=self._user_id,
            agent_id=self._agent_id,
            max_user_turns=max_user_turns,
        )
        return [row for row in (rows or []) if self._trace_visible(row)]

    async def append_trace(
        self,
        messages: Sequence[Any],
        *,
        request_id: str,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str | None = None,
        workspace_id: str | None = None,
        branch: str | None = None,
    ) -> None:
        """Append raw trace rows for this lease only.

        Trace rows are raw conversation, so this is a write path: the lease is
        rechecked around the call and every message must belong to the leased
        user and session. Nothing outside the lease may be recorded.
        """
        self._scope(user_id)
        if agent_id not in (self._agent_id, "*"):
            self._deny("scope")
        if session_id not in (None, self._session_id):
            self._deny("scope")
        for row in messages or []:
            if not self._trace_visible(row):
                self._deny("scope")
        await self._invoke(
            "append_trace",
            list(messages or []),
            request_id=request_id,
            user_id=self._user_id,
            agent_id=self._agent_id,
            session_id=self._session_id,
            workspace_id=workspace_id,
            branch=branch,
        )

    def _trace_visible(self, row: Any) -> bool:
        """A trace row may be read or written only inside this lease's scope."""
        if row is None:
            return False
        owner = getattr(row, "user_id", None)
        session = getattr(row, "session_id", None)
        return not (
            (owner is not None and owner != self._user_id)
            or (session is not None and session != self._session_id)
        )
