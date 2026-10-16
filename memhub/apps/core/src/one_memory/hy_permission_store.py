"""Operation-bound read/write capability for the pinned native HY chat vector store.

No semantic storage or extraction lives here. Pre-existing nodes require a full
native-node attestation with complete approved Evidence provenance; known content
alone cannot authorize tags, custom fields, vectors or links. Unknown nodes stay
in the same HY backend but are not exposed to this operation.

The caller must exclusively own the native backend's writes during execution:
its API has no compare-and-swap, and a swallowed get_by_id error can look missing.
This component cannot make that check/upsert interval atomic. Factory integration
must replace client/registry/both writers/cached-reader references, bind physical
LLM/embedding sends to the lease AND this denial latch, and guard publication.
Coding rows go through hy_permission_coding_store instead; graph, entities,
digest, sweeper and summary remain unsupported.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, NoReturn

from hy_memory.models.memory import (  # type: ignore[import-untyped]
    MemoryLayer,
    MemoryNode,
    MemoryStatus,
)

from one_memory.hy_step_timing import HyStepTimer
from one_memory.model_processing_permission_guard import (
    ModelProcessingPermissionLease,
    permission_error,
)

_WRITE_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)


def _native_write_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _WRITE_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _WRITE_LOCKS[loop] = lock
    return lock


@asynccontextmanager
async def native_store_read_slot() -> AsyncIterator[bool]:
    """Admit one native HY read only while no write is active.

    Chroma's native reader may retain the GIL after an asyncio timeout. Running
    it concurrently with HY's multi-step store writes has deadlocked the Core
    under sustained imports. Reads therefore share the physical store lock,
    but never wait behind a writer: host recall can immediately use the local
    governance ledger instead.
    """
    lock = _native_write_lock()
    if lock.locked():
        yield False
        return
    await lock.acquire()
    try:
        yield True
    finally:
        lock.release()


@dataclass(frozen=True)
class HyNodePermission:
    """Trusted caller's full-node attestation, not permission inferred from an ID."""

    fingerprint: str
    evidence_ids: tuple[str, ...]


def hy_node_fingerprint(node: MemoryNode) -> str:
    """Hash the complete native serialized node, including vectors and metadata.

    Timestamp precision follows HY's to_dict contract (integer Unix seconds).
    Attest the node read from the backend, not an unpersisted input object.
    """
    if type(node) is not MemoryNode:
        raise permission_error("hy_store_node_shape")
    return sha256(
        json.dumps(node.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


_CHAT_LAYERS = frozenset(
    {
        MemoryLayer.L0_BASIC_INFO,
        MemoryLayer.L1_RAW,
        MemoryLayer.L2_FACT,
        MemoryLayer.L4_IDENTITY,
        MemoryLayer.L7_INTENTION,
    }
)
_UPDATE_FIELDS = frozenset(
    {
        "content",
        "embedding",
        "tags",
        "tag_ners",
        "layer",
        "memory_at",
        "gmt_modified",
        "is_latest",
        "status",
        "supersedes",
        "superseded_by",
        "source_raw_memory_id",
        # SDK 2.0.2 sends both of these on an in-place reconcile UPDATE.
        # They are metadata only and must never carry identity, permission or
        # tenant state: `owner` is a provenance label whose only values are
        # "user" / "agent" (who said this line), and `custom` is the SDK's own
        # free-form notes field. Neither may hold a user_id, scope, grant or
        # tenancy marker, so allowing the field names does not widen the
        # cross-user boundary that the store still enforces per node.
        "owner",
        "custom",
    }
)
_DEDUP_AUDIT_FIELDS = frozenset({"shadow_reason", "dedup_kept_id"})
_DIGEST_COUNTER_FIELD = "s2_evidence_count"


def _write_view(node: MemoryNode) -> dict[str, Any]:
    """Only native Chroma's empty-string and float32 storage normalization."""
    payload: dict[str, Any] = node.to_dict()
    payload["source_session_id"] = payload["source_session_id"] or ""
    # The pinned public SDK serializes tag_ners as JSON, but its Chroma read path omits this
    # field from the JSON decode list. Decode only the exact list-of-strings
    # shape that the write path produced; malformed values still fail the
    # complete-node comparison below.
    tag_ners = payload.get("tag_ners")
    if (
        isinstance(tag_ners, list)
        and tag_ners
        and all(isinstance(value, str) and len(value) == 1 for value in tag_ners)
        and tag_ners[0] == "["
        and tag_ners[-1] == "]"
    ):
        tag_ners = "".join(tag_ners)
    if isinstance(tag_ners, str):
        try:
            decoded = json.loads(tag_ners)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, list) and all(isinstance(value, str) for value in decoded):
            payload["tag_ners"] = decoded
    if payload.get("tag_ners") in ([], ["[", "]"]):
        payload["tag_ners"] = []
    # HY Chroma encodes None as ""; these JSON fields keep "" when json.loads
    # fails (vector_store_chroma.py:741-788). Only an absent link is equivalent.
    for field in ("supersedes", "superseded_by"):
        if payload[field] == "":
            payload[field] = None
    if payload["embedding"] is not None:
        payload["embedding"] = [
            struct.unpack("f", struct.pack("f", n))[0] for n in payload["embedding"]
        ]
        if not all(math.isfinite(n) for n in payload["embedding"]):
            raise ValueError("nonfinite_embedding")
    return payload


def _same_written_vector(actual: list[float] | None, expected: list[float] | None) -> bool:
    if actual == expected:
        return True
    if actual is None or expected is None or len(actual) != len(expected):
        return False
    # Chroma's cosine index can renormalize float32 vectors on readback. Bound
    # only this write roundtrip; the saved full-node proof uses exact readback.
    return all(
        math.isclose(left, right, rel_tol=4 * 2**-23, abs_tol=0.0)
        for left, right in zip(actual, expected, strict=True)
    )


_NATIVE_WRITES = frozenset(
    {"upsert", "upsert_batch", "update_payload", "update_embedding", "delete"}
)


class HyPermissionGraphStore:
    """Lease-gated graph store for HY System 2's digest transaction.

    The native graph is deliberately absent from ordinary ingest leases. Digest
    needs it to write L6 schemas, but every operation must remain inside the
    same frozen permission lease rather than exposing the raw store.
    """

    def __init__(self, native: Any, recheck: Any) -> None:
        self._native = native
        self._recheck = recheck
        self._closed = False

    def recheck(self) -> None:
        if self._closed:
            raise permission_error("hy_graph_store_closed")
        self._recheck()

    async def close(self) -> None:
        self._closed = True

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name != "_run":
            raise AttributeError(name)
        attribute = getattr(self._native, name)
        if not callable(attribute):
            return attribute

        async def guarded(*args: Any, **kwargs: Any) -> Any:
            self.recheck()
            result = await attribute(*args, **kwargs)
            self.recheck()
            return result

        return guarded


class HyPermissionVectorStore:
    """Closed, task-exclusive adapter; initialize/close never own the native store.

    node_permissions may attest additional native raw/profile nodes not represented
    in known_memories. Every attestation must cover all its contributors, and all
    Evidence IDs must already be in this lease. Known entries also constrain exact
    content/owner. Without a full attestation even a known entry is denied.

    written_nodes records successful writes with the entire lease footprint, a
    conservative provenance bound, not a claim of semantic citation accuracy.
    These transient fingerprints must be persisted by the integrating factory if
    future operations are to reuse raw/profile nodes. No content is stored here.
    """

    __slots__ = (
        "__native",
        "__lease",
        "__agent_id",
        "__session_id",
        "__cross_evidence_scope",
        "__user_id",
        "__known",
        "__permissions",
        "__written",
        "__evidence_ids",
        "__deleted",
        "__shadowed",
        "__superseded",
        "__heads",
        "__lock",
        "__closed",
        "__failure",
        "__step_timer",
        "__refresh_permission",
    )

    def __init__(
        self,
        native: Any,
        lease: ModelProcessingPermissionLease,
        *,
        agent_id: str,
        session_id: str,
        cross_evidence_scope: bool = False,
        node_permissions: Mapping[str, HyNodePermission] | None = None,
        refresh_permission: Callable[[str], Awaitable[HyNodePermission | None]] | None = None,
        step_timer: HyStepTimer | None = None,
    ) -> None:
        self.__step_timer = step_timer
        self.__native = native
        self.__lease = lease
        self.__agent_id = agent_id
        self.__session_id = session_id
        self.__cross_evidence_scope = cross_evidence_scope
        self.__user_id = lease.user_id
        self.__failure: str | None = None
        self.__closed = False
        self.__lock = asyncio.Lock()
        self.__deleted: set[str] = set()
        self.__written: dict[str, HyNodePermission] = {}
        self.__shadowed: set[str] = set()
        self.__superseded: dict[str, tuple[str, ...]] = {}
        self.__heads: dict[str, tuple[str, ...]] = {}
        self.__refresh_permission = refresh_permission
        self.recheck()
        self.__evidence_ids = tuple(record.evidence_id for record in lease.records)
        current = lease.source_context.current if lease.source_context else lease.records
        conversation_mismatch = any(r.conversation_id != session_id for r in current)
        if (
            not lease.task_id
            or not agent_id
            or agent_id == "*"
            or not session_id
            or not current
            or (conversation_mismatch and not cross_evidence_scope)
            or not self.__evidence_ids
            or len(set(self.__evidence_ids)) != len(lease.records)
            or any(r.user_id != self.__user_id for r in lease.records)
        ):
            self.__deny("scope")
        known = lease.source_context.known_memories if lease.source_context else ()
        self.__known = {item.hy_memory_id: item for item in known}
        if len(self.__known) != len(known):
            self.__deny("provenance")
        self.__permissions = dict(node_permissions or {})
        for key, proof in self.__permissions.items():
            if (
                not key
                or not isinstance(proof, HyNodePermission)
                or len(proof.fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in proof.fingerprint)
                or not isinstance(proof.evidence_ids, tuple)
                or not proof.evidence_ids
                or len(set(proof.evidence_ids)) != len(proof.evidence_ids)
                or not set(proof.evidence_ids) <= set(self.__evidence_ids)
            ):
                self.__deny("provenance")
        for key, item in self.__known.items():
            if not item.evidence_ids or not set(item.evidence_ids) <= set(self.__evidence_ids):
                self.__deny("provenance")
            known_proof = self.__permissions.get(key)
            if known_proof and not set(item.evidence_ids) <= set(known_proof.evidence_ids):
                self.__deny("provenance")

    @property
    def written_nodes(self) -> Mapping[str, HyNodePermission]:
        """Latest verified write proofs; each access returns a new immutable snapshot.

        Export after SDK writes have drained and before close/governance projection.
        Merge these rows into the lineage ledger, then remove deleted_node_ids;
        this write delta does not include untouched pre-existing permissions.
        """
        self.recheck()
        return MappingProxyType(dict(self.__written))

    @property
    def deleted_node_ids(self) -> frozenset[str]:
        """Verified native deletions in this operation, for lineage-ledger cleanup."""
        self.recheck()
        return frozenset(self.__deleted)

    @property
    def retired_node_ids(self) -> frozenset[str]:
        """Nodes HY deleted or shadowed (its logical delete) in this operation."""
        self.recheck()
        return frozenset(self.__deleted | self.__shadowed)

    @property
    def superseded_nodes(self) -> Mapping[str, tuple[str, ...]]:
        """Nodes HY marked superseded in this operation, with what replaced them.

        HY writes the two directions separately and may fail either one, so a
        replacement counts whether it was recorded on the old node or only on
        the new head.
        """
        self.recheck()
        result = {}
        for key, by in self.__superseded.items():
            heads = set(by) | {head for head, olds in self.__heads.items() if key in olds}
            heads -= self.__deleted | {key}
            if key not in self.__deleted and heads:
                result[key] = tuple(sorted(heads))
        return MappingProxyType(result)

    @property
    def supports_fulltext(self) -> bool:
        return False

    def recheck(self) -> None:
        """Must also gate transport/publication after HY swallows a store failure."""
        self.__assert_open()
        try:
            self.__lease.recheck()
        except Exception as error:
            code = str(error)
            self.__failure = (
                code
                if code.startswith("model_content_") and len(code) <= 100
                else "model_content_hy_store_lease_failed"
            )
            raise permission_error(self.__failure) from None

    def __assert_open(self) -> None:
        if self.__closed:
            self.__deny("closed")
        if self.__failure:
            raise permission_error(self.__failure)

    async def initialize(self) -> None:
        self.recheck()

    async def close(self) -> None:
        """Revoke this wrapper locally without rechecking context or closing HY.

        Cleanup must remain possible after governance projection changes the
        lease's known-memory revisions, or after permission has been revoked.
        """
        async with self.__lock:
            self.__closed = True
            self.__permissions.clear()
            self.__written.clear()

    async def has_tag_embedding(self, user_id: str, tag: str) -> bool:
        self.recheck()
        # Native chat calls this best-effort helper. NotImplemented prevents it
        # from embedding tags or falling through to an unguarded entity store.
        raise NotImplementedError("hy_permission_store_tag_index_unsupported")

    def __deny(self, suffix: str) -> NoReturn:
        self.__failure = self.__failure or f"model_content_hy_store_{suffix}"
        raise permission_error(self.__failure)

    async def __invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        writing = method in _NATIVE_WRITES
        self.recheck() if writing else self.__assert_open()
        step = "vector_store_write" if writing else "vector_store_read"
        first = args[0] if args else None
        try:
            with (
                self.__step_timer.step(step, items=len(first) if isinstance(first, list) else 1)
                if self.__step_timer is not None
                else nullcontext()
            ):
                result = await getattr(self.__native, method)(*args, **kwargs)
        except Exception:
            self.__deny("backend_failed")
        self.recheck() if writing else self.__assert_open()
        return result

    async def __read_native(self, key: str) -> MemoryNode | None:
        value = await self.__invoke("get_by_id", key)
        if value is not None and (type(value) is not MemoryNode or value.node_id != key):
            self.__deny("node_shape")
        return deepcopy(value)

    def __fingerprint(self, node: MemoryNode) -> str:
        try:
            return hy_node_fingerprint(node)
        except (TypeError, ValueError, AttributeError, OverflowError):
            self.__deny("node_shape")

    def __write_view(self, node: MemoryNode) -> dict[str, Any]:
        self.__fingerprint(node)
        try:
            return _write_view(node)
        except (TypeError, ValueError, OverflowError, struct.error):
            self.__deny("node_shape")

    def __in_scope(self, node: MemoryNode) -> bool:
        if node.user_id != self.__user_id or node.layer not in _CHAT_LAYERS:
            return False
        if node.layer is MemoryLayer.L0_BASIC_INFO:
            # Public SDK 2.0.2 migrates verified legacy agent-level profiles into one
            # user-level chain. Existing nodes still require a full proof in
            # __load; accepting their historical agent id does not authorize
            # unproven cross-agent memories.
            return True
        if self.__cross_evidence_scope:
            return bool(node.agent_id == self.__agent_id)
        return bool(node.agent_id == self.__agent_id and node.session_id == self.__session_id)

    def __new_node_in_scope(self, node: MemoryNode) -> bool:
        if not self.__in_scope(node):
            return False
        if node.layer is MemoryLayer.L0_BASIC_INFO:
            return node.agent_id in ("", self.__agent_id) and node.session_id in (
                "",
                self.__session_id,
            )
        return True

    def __references(self, node: MemoryNode) -> list[str]:
        refs = []
        for field in ("supersedes", "superseded_by", "evidence_chain"):
            values = getattr(node, field)
            if field != "evidence_chain" and values == "":
                continue
            if values is not None:
                if not isinstance(values, list) or any(
                    not isinstance(v, str) or not v for v in values
                ):
                    self.__deny("node_shape")
                refs.extend(values)
        if node.source_raw_memory_id is not None:
            if not isinstance(node.source_raw_memory_id, str) or not node.source_raw_memory_id:
                self.__deny("node_shape")
            refs.append(node.source_raw_memory_id)
        return refs

    async def __load(
        self,
        key: str,
        first: MemoryNode | None = None,
        *,
        allow_deleted_links: bool = False,
    ) -> MemoryNode | None:
        pending = [key]
        checked: dict[str, MemoryNode] = {}
        while pending:
            current = pending.pop()
            if current in checked:
                continue
            # HY dedup deletes head then ancestors. Only this operation's
            # verified tombstones may terminate a closure during deletion.
            if allow_deleted_links and current != key and current in self.__deleted:
                continue
            proof = self.__permissions.get(current)
            if proof is None or current in self.__deleted:
                return None
            value = (
                first if current == key and first is not None else await self.__read_native(current)
            )
            if value is None or not self.__in_scope(value):
                return None
            if self.__fingerprint(value) != proof.fingerprint:
                refreshed = (
                    await self.__refresh_permission(current)
                    if self.__refresh_permission is not None
                    else None
                )
                if refreshed is None or self.__fingerprint(value) != refreshed.fingerprint:
                    self.__deny("node_changed")
                self.__permissions[current] = refreshed
                proof = refreshed
            known = self.__known.get(current) if current not in self.__written else None
            if known and (value.content != known.content or value.owner != known.owner):
                self.__deny("node_changed")
            checked[current] = value
            pending.extend(self.__references(value))
        return deepcopy(checked[key])

    async def get_by_id(self, node_id: str) -> MemoryNode | None:
        async with self.__lock:
            self.recheck()
            result = await self.__load(node_id)
            self.recheck()
            return result

    async def get_by_ids(self, node_ids: list[str]) -> list[MemoryNode]:
        async with self.__lock:
            self.recheck()
            result = [
                n for key in dict.fromkeys(node_ids) if (n := await self.__load(key)) is not None
            ]
            self.recheck()
            return result

    async def get_embeddings(self, node_ids: list[str]) -> dict[str, list[float]]:
        return {
            n.node_id: list(n.embedding)
            for n in await self.get_by_ids(node_ids)
            if n.embedding is not None
        }

    def __scope(self, user_id: str | None, agent_id: str | None) -> None:
        if user_id not in (None, self.__user_id) or agent_id not in (None, self.__agent_id):
            self.__deny("scope")

    async def list_by_user(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10000,
        status_filter: list[Any] | None = None,
        layers: list[Any] | None = None,
        user_ids: list[str] | None = None,
        agent_ids: list[str] | None = None,
        only_latest: bool = False,
        with_vectors: bool = True,
        allow_raw_layer: bool = False,
    ) -> list[MemoryNode]:
        async with self.__lock:
            self.recheck()
            effective_users = user_ids or ([user_id] if user_id else [])
            effective_agents = agent_ids or ([agent_id] if agent_id else [])
            if effective_users != [self.__user_id] or any(
                value != self.__agent_id for value in effective_agents
            ):
                self.__deny("scope")
            rows = await self.__invoke(
                "list_by_user",
                user_id=self.__user_id,
                agent_id=agent_id,
                limit=limit,
                status_filter=status_filter,
                layers=layers,
                user_ids=[self.__user_id],
                agent_ids=effective_agents or None,
                only_latest=only_latest,
                with_vectors=with_vectors,
                allow_raw_layer=allow_raw_layer,
            )
            result = []
            for row in rows:
                if type(row) is not MemoryNode:
                    self.__deny("node_shape")
                value = await self.__load(row.node_id)
                if (
                    value is not None
                    and (not layers or value.layer in layers or value.layer.value in layers)
                    and (
                        not status_filter
                        or value.status in status_filter
                        or value.status.value in status_filter
                    )
                ):
                    result.append(value)
            self.recheck()
            return result[:limit]

    async def search(
        self,
        query_embedding: list[float],
        isolation_key: str = "",
        isolation_keys: list[str] | None = None,
        user_id: str | None = None,
        user_ids: list[str] | None = None,
        agent_ids: list[str] | None = None,
        layers: list[Any] | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        status_filter: list[Any] | None = None,
        only_latest: bool = True,
        tags_match_any: list[str] | None = None,
        created_after: float | None = None,
    ) -> list[dict[str, Any]]:
        async with self.__lock:
            self.recheck()
            self.__scope(user_id, None)
            key = MemoryNode.build_isolation_key(self.__user_id, self.__agent_id, self.__session_id)
            if (
                any(uid != self.__user_id for uid in user_ids or [])
                or any(aid != self.__agent_id for aid in agent_ids or [])
                or isolation_key not in ("", key)
                or any(ik != key for ik in isolation_keys or [])
            ):
                self.__deny("scope")
            hits = await self.__invoke(
                "search",
                query_embedding=deepcopy(query_embedding),
                isolation_key=isolation_key,
                isolation_keys=isolation_keys,
                user_id=self.__user_id,
                user_ids=[self.__user_id],
                agent_ids=[self.__agent_id],
                layers=layers,
                limit=limit,
                score_threshold=score_threshold,
                status_filter=status_filter,
                only_latest=only_latest,
                tags_match_any=tags_match_any,
                created_after=created_after,
            )
            result = []
            for hit in hits:
                value = await self.__load(hit["node_id"])
                if value is None:
                    continue
                score = float(hit["score"])
                if not math.isfinite(score):
                    self.__deny("node_shape")
                # Do not forward arbitrary backend hit metadata. Rehydrate the
                # full fingerprinted node; native search often omits embedding.
                result.append({"node_id": value.node_id, "node": value, "score": score})
            self.recheck()
            return result[:limit]

    async def __target(
        self,
        key: str,
        first: MemoryNode | None = None,
        *,
        allow_deleted_links: bool = False,
    ) -> MemoryNode:
        value = await self.__load(key, first, allow_deleted_links=allow_deleted_links)
        if value is None:
            self.__deny("target_denied")
        return value

    async def __check_links(self, value: MemoryNode) -> None:
        for ref in self.__references(value):
            if await self.__load(ref) is None:
                self.__deny("link_denied")

    async def __plan_upsert(self, value: MemoryNode) -> MemoryNode:
        value = deepcopy(value)
        if type(value) is not MemoryNode or not value.node_id or not self.__in_scope(value):
            self.__deny("scope")
        existing = await self.__read_native(value.node_id)
        if existing is not None:
            await self.__target(value.node_id, existing)
        elif (
            not self.__new_node_in_scope(value)
            or value.node_id in self.__permissions
            or value.node_id in self.__known
            or value.node_id in self.__deleted
        ):
            self.__deny("scope")
        await self.__check_links(value)
        self.__write_view(value)
        return value

    async def __capture(
        self,
        expected: MemoryNode,
        *,
        allow_native_modified_time: bool = False,
    ) -> None:
        actual = await self.__read_native(expected.node_id)
        if actual is None:
            self.__deny("write_mismatch")
        actual_view, expected_view = self.__write_view(actual), self.__write_view(expected)
        if allow_native_modified_time:
            if actual.gmt_modified < expected.gmt_modified:
                self.__deny("write_mismatch")
            actual_view["gmt_modified"] = expected_view["gmt_modified"]
        actual_vector = actual_view.pop("embedding")
        expected_vector = expected_view.pop("embedding")
        if actual_view != expected_view or not _same_written_vector(actual_vector, expected_vector):
            self.__deny("write_mismatch")
        proof = HyNodePermission(self.__fingerprint(actual), self.__evidence_ids)
        self.__permissions[actual.node_id] = proof
        self.__written[actual.node_id] = proof
        status = getattr(actual.status, "value", actual.status)
        if status == MemoryStatus.SHADOW.value:
            self.__shadowed.add(actual.node_id)
        else:
            self.__shadowed.discard(actual.node_id)
        if status == MemoryStatus.SUPERSEDED.value:
            by = actual.superseded_by
            self.__superseded[actual.node_id] = tuple(by) if isinstance(by, list) else ()
        else:
            self.__superseded.pop(actual.node_id, None)
        olds = actual.supersedes
        if isinstance(olds, list) and olds:
            self.__heads[actual.node_id] = tuple(olds)
        else:
            self.__heads.pop(actual.node_id, None)

    async def upsert(self, node: MemoryNode) -> str:
        async with _native_write_lock(), self.__lock:
            self.recheck()
            expected = await self.__plan_upsert(node)
            result = await self.__invoke("upsert", deepcopy(expected))
            if result != expected.node_id:
                self.__deny("write_mismatch")
            await self.__capture(expected)
            return str(result)

    async def upsert_batch(self, nodes: list[MemoryNode]) -> list[str]:
        async with _native_write_lock(), self.__lock:
            self.recheck()
            if len({n.node_id for n in nodes}) != len(nodes):
                self.__deny("node_shape")
            # No forward links to uncommitted batch members. Preflight every
            # existing target before the backend can make a partial write.
            planned = [await self.__plan_upsert(n) for n in nodes]
            ids = [n.node_id for n in planned]
            result = await self.__invoke("upsert_batch", deepcopy(planned))
            if result != ids:
                self.__deny("write_mismatch")
            for expected in planned:
                await self.__capture(expected)
            return ids

    async def update_payload(self, node_id: str, updates: dict[str, Any]) -> bool:
        async with _native_write_lock(), self.__lock:
            self.recheck()
            current = await self.__target(node_id)
            updates = deepcopy(updates)
            dedup_audit = False
            if set(updates) & _DEDUP_AUDIT_FIELDS:
                if (
                    not set(updates) <= (_UPDATE_FIELDS | _DEDUP_AUDIT_FIELDS)
                    or updates.get("status") != MemoryStatus.SHADOW.value
                    or updates.get("is_latest") is not False
                    or not isinstance(updates.get("shadow_reason"), str)
                    or not updates["shadow_reason"].startswith("dedup:")
                    or not isinstance(updates.get("dedup_kept_id"), str)
                    or not updates["dedup_kept_id"]
                ):
                    self.__deny("update_fields")
                await self.__target(updates["dedup_kept_id"])
                custom = deepcopy(current.custom)
                custom["hy_dedup"] = {
                    "reason": updates.pop("shadow_reason"),
                    "kept_id": updates.pop("dedup_kept_id"),
                }
                updates["custom"] = custom
                dedup_audit = True
            allowed = _UPDATE_FIELDS | ({"custom"} if dedup_audit else set())
            if self.__cross_evidence_scope and set(updates) == {"custom"}:
                custom = updates.get("custom")
                if not isinstance(custom, dict):
                    self.__deny("update_fields")
                current_custom = current.custom if isinstance(current.custom, dict) else {}
                previous = current_custom.get(_DIGEST_COUNTER_FIELD, 0)
                proposed = custom.get(_DIGEST_COUNTER_FIELD)
                if (
                    not isinstance(previous, int)
                    or isinstance(previous, bool)
                    or not isinstance(proposed, int)
                    or isinstance(proposed, bool)
                    or proposed != previous + 1
                    or set(custom) != set(current_custom) | {_DIGEST_COUNTER_FIELD}
                    or any(custom[key] != value for key, value in current_custom.items())
                ):
                    self.__deny("update_fields")
                allowed |= {"custom"}
            # `custom` is an allowed field name (SDK 2.0.2 sends it on a
            # reconcile UPDATE), but the System 2 evidence counter it can hold
            # stays digest-only: only the digest scope may advance it, and only
            # by one per write. Letting the name through must not let an
            # ordinary write move that counter.
            if not self.__cross_evidence_scope and _DIGEST_COUNTER_FIELD in (
                updates.get("custom") or {}
            ):
                self.__deny("update_fields")
            if not set(updates) <= allowed:
                self.__deny("update_fields")
            try:
                if "layer" in updates:
                    MemoryLayer(updates["layer"])
                if "status" in updates:
                    MemoryStatus(updates["status"])
                if "tag_ners" in updates and (
                    not isinstance(updates["tag_ners"], list)
                    or any(not isinstance(value, str) for value in updates["tag_ners"])
                ):
                    self.__deny("node_shape")
                expected = MemoryNode.from_dict({**current.to_dict(), **updates})
                self.__fingerprint(expected)
            except (TypeError, ValueError):
                self.__deny("node_shape")
            if not self.__in_scope(expected):
                self.__deny("scope")
            await self.__check_links(expected)
            self.__write_view(expected)
            if await self.__invoke("update_payload", node_id, updates) is not True:
                self.__deny("write_failed")
            await self.__capture(expected, allow_native_modified_time=True)
            return True

    async def update_embedding(self, node_id: str, embedding: list[float]) -> bool:
        async with _native_write_lock(), self.__lock:
            self.recheck()
            expected = await self.__target(node_id)
            expected.embedding = deepcopy(embedding)
            self.__write_view(expected)
            if await self.__invoke("update_embedding", node_id, deepcopy(embedding)) is not True:
                self.__deny("write_failed")
            await self.__capture(expected, allow_native_modified_time=True)
            return True

    async def delete(self, node_id: str) -> bool:
        async with _native_write_lock(), self.__lock:
            self.recheck()
            await self.__target(node_id, allow_deleted_links=True)
            if await self.__invoke("delete", node_id) is not True:
                self.__deny("write_failed")
            if await self.__read_native(node_id) is not None:
                self.__deny("write_mismatch")
            self.__deleted.add(node_id)
            self.__permissions.pop(node_id, None)
            self.__written.pop(node_id, None)
            self.__superseded.pop(node_id, None)
            self.__heads.pop(node_id, None)
            return True
