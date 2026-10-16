"""Native HY node/algorithm tests with synthetic data and an in-memory backend only."""

import asyncio
import math
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from hy_memory.agent.tools.basic_profile import upsert_basic_profile
from hy_memory.data.vector_store_chroma import ChromaVectorStore
from hy_memory.models.memory import MemoryLayer, MemoryNode, MemoryStatus
from hy_memory.pipelines._retrieval.dedup import DedupItem, execute_dedup
from hy_memory.pipelines._retrieval.evolution import _trace_full_chain
from hy_memory.pipelines.writer import MemoryWriter

from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.hy_permission_store import (
    HyNodePermission,
    HyPermissionVectorStore,
    _native_write_lock,
    hy_node_fingerprint,
    native_store_read_slot,
)
from one_memory.hy_source_context import HySourceContext, HySupportedMemory
from one_memory.imports.models import EvidenceContentKind, EvidenceRecord, WrittenBy


def test_native_semantic_read_yields_immediately_to_an_active_write() -> None:
    async def scenario() -> tuple[bool, bool]:
        async with _native_write_lock(), native_store_read_slot() as during_write:
            pass
        async with native_store_read_slot() as after_write:
            pass
        return during_write, after_write

    assert asyncio.run(scenario()) == (False, True)


def evidence(key):
    now = datetime(2026, 9, 13, tzinfo=UTC)
    return EvidenceRecord(
        evidence_id=key,
        user_id="local",
        source_id="cursor",
        workspace=None,
        project=None,
        conversation_id="session",
        message_id=key,
        correlation_id=None,
        source_ordinal=None,
        content_kind=EvidenceContentKind.RAW_CONVERSATION,
        role="user",
        written_by=WrittenBy.PERSON,
        created_at=now,
        verbatim=f"Synthetic evidence {key}",
        origin_meta={},
        content_hash="a" * 64,
        revision=1,
        imported_at=now,
        updated_at=now,
    )


def node(key, **kwargs):
    return MemoryNode(
        **{
            "node_id": key,
            "user_id": "local",
            "agent_id": "agent",
            "session_id": "session",
            "content": f"Synthetic {key}",
            "owner": "user",
            "layer": MemoryLayer.L2_FACT,
            "embedding": [1.0, 0.0],
            **kwargs,
        }
    )


class Lease:
    def __init__(self, known=()):
        self.task_id = "task"
        self.receipt = SimpleNamespace(user_id="local")
        self.records = (evidence("current"), evidence("support"))
        self.source_context = HySourceContext(
            current=self.records[:1],
            supporting=self.records[1:],
            known_memories=known,
        )
        self.revoked = False
        self.checks = 0

    def recheck(self):
        self.checks += 1
        if self.revoked:
            raise HyMemoryBridgeError("model_content_permission_revoked")
        return self.receipt


class Backend:
    def __init__(self, *nodes):
        self.nodes = {n.node_id: deepcopy(n) for n in nodes}
        self.calls = []
        self.after_read = lambda: None
        self.corrupt_write = lambda _: None

    async def get_by_id(self, key):
        self.calls.append(("get", key))
        result = deepcopy(self.nodes.get(key))
        self.after_read()
        return result

    async def list_by_user(self, **kwargs):
        self.calls.append(("list", kwargs))
        # Deliberately return foreign/unsupported rows to exercise defense in depth.
        return deepcopy(list(self.nodes.values()))

    async def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        hits = []
        for n in self.nodes.values():
            candidate = deepcopy(n)
            candidate.embedding = None  # Native Chroma search omits the vector.
            hits.append(
                {
                    "node_id": n.node_id,
                    "node": candidate,
                    "score": 0.9,
                    "untrusted_extra": "DENIED_HIT_METADATA",
                }
            )
        return hits

    async def upsert(self, value):
        self.calls.append(("upsert", value.node_id))
        self.nodes[value.node_id] = deepcopy(value)
        self.corrupt_write(self.nodes[value.node_id])
        return value.node_id

    async def upsert_batch(self, values):
        return [await self.upsert(n) for n in values]

    async def update_payload(self, key, updates):
        self.calls.append(("update", key, deepcopy(updates)))
        self.nodes[key] = MemoryNode.from_dict({**self.nodes[key].to_dict(), **deepcopy(updates)})
        self.corrupt_write(self.nodes[key])
        return True

    async def update_embedding(self, key, embedding):
        self.calls.append(("embedding", key))
        self.nodes[key].embedding = deepcopy(embedding)
        return True

    async def delete(self, key):
        self.calls.append(("delete", key))
        self.nodes.pop(key, None)
        return True


def permission(value, ids=("current", "support")):
    return HyNodePermission(fingerprint=hy_node_fingerprint(value), evidence_ids=ids)


def wrapper(backend, *, lease=None, approved=(), refresh_permission=None):
    return HyPermissionVectorStore(
        backend,
        lease or Lease(),
        agent_id="agent",
        session_id="session",
        node_permissions={n.node_id: permission(n) for n in approved},
        refresh_permission=refresh_permission,
    )


def test_cross_evidence_scope_is_opt_in_and_default_leases_stay_session_bound():
    lease = Lease()
    mismatched = lease.records[1].model_copy(update={"conversation_id": "other-conversation"})
    lease.records = (lease.records[0], mismatched)
    lease.source_context = HySourceContext(current=lease.records)

    with pytest.raises(HyMemoryBridgeError, match="scope"):
        HyPermissionVectorStore(
            Backend(),
            lease,
            agent_id="agent",
            session_id="session",
        )

    store = HyPermissionVectorStore(
        Backend(),
        lease,
        agent_id="agent",
        session_id="digest:task",
        cross_evidence_scope=True,
    )
    assert store is not None


def test_cross_evidence_scope_does_not_relax_user_or_evidence_integrity():
    lease = Lease()
    foreign = lease.records[1].model_copy(update={"user_id": "someone-else"})
    lease.records = (lease.records[0], foreign)
    with pytest.raises(HyMemoryBridgeError, match="scope"):
        HyPermissionVectorStore(
            Backend(),
            lease,
            agent_id="agent",
            session_id="digest:task",
            cross_evidence_scope=True,
        )

    duplicate = Lease()
    duplicate.records = (duplicate.records[0], duplicate.records[0])
    with pytest.raises(HyMemoryBridgeError, match="scope"):
        HyPermissionVectorStore(
            Backend(),
            duplicate,
            agent_id="agent",
            session_id="digest:task",
            cross_evidence_scope=True,
        )


def test_cross_evidence_scope_allows_digest_to_read_other_sessions_of_the_same_agent():
    lease = Lease()
    lease.source_context = HySourceContext(current=lease.records)
    existing = node("digest-source", session_id="note:one")
    backend = Backend(existing)
    approved = {existing.node_id: permission(existing)}
    regular = wrapper(backend, approved=(existing,))
    digest = HyPermissionVectorStore(
        backend,
        lease,
        agent_id="agent",
        session_id="digest:task",
        cross_evidence_scope=True,
        node_permissions=approved,
    )

    async def scenario():
        assert await regular.get_by_id(existing.node_id) is None
        found = await digest.get_by_id(existing.node_id)
        assert found is not None and found.node_id == existing.node_id

    asyncio.run(scenario())


def test_only_digest_scope_may_advance_the_system2_evidence_counter_once():
    approved = node("digest-source", custom={})
    lease = Lease()
    lease.source_context = HySourceContext(current=lease.records)
    backend = Backend(approved)
    permissions = {approved.node_id: permission(approved)}
    regular = wrapper(backend, approved=(approved,))
    digest = HyPermissionVectorStore(
        backend,
        lease,
        agent_id="agent",
        session_id="digest:task",
        cross_evidence_scope=True,
        node_permissions=permissions,
    )

    async def scenario():
        with pytest.raises(HyMemoryBridgeError, match="update_fields"):
            await regular.update_payload(approved.node_id, {"custom": {"s2_evidence_count": 1}})
        assert await digest.update_payload(approved.node_id, {"custom": {"s2_evidence_count": 1}})
        with pytest.raises(HyMemoryBridgeError, match="update_fields"):
            await digest.update_payload(
                approved.node_id,
                {"custom": {"s2_evidence_count": 3, "extra": "DENIED"}},
            )

    asyncio.run(scenario())


def test_known_content_alone_cannot_authorize_native_extras_or_mutation():
    allowed = node("known", tags=["DENIED_TAG"], custom={"note": "DENIED_CUSTOM"})
    known = HySupportedMemory(
        hy_memory_id=allowed.node_id,
        governed_memory_id="governed",
        revision=1,
        content=allowed.content,
        owner="user",
        evidence_ids=("current",),
    )
    backend = Backend(allowed)
    store = wrapper(backend, lease=Lease((known,)))

    async def scenario():
        assert await store.get_by_id("known") is None
        assert await store.list_by_user("local") == []
        with pytest.raises(HyMemoryBridgeError, match="target_denied"):
            await store.update_payload("known", {"content": "changed"})

    asyncio.run(scenario())
    assert backend.nodes["known"].custom == allowed.custom
    assert not any(call[0] == "update" for call in backend.calls)


def test_all_native_read_shapes_keep_allowed_ids_and_filter_foreign_payloads():
    allowed, denied = node("allowed"), node("denied", content="DENIED_CONTENT")
    foreign = node("foreign", user_id="someone-else")
    backend = Backend(allowed, denied, foreign)
    store = wrapper(backend, approved=(allowed, foreign))

    async def scenario():
        assert await store.get_by_id("denied") is None
        assert [n.node_id for n in await store.get_by_ids(["allowed", "denied"])] == ["allowed"]
        assert await store.get_embeddings(["denied", "allowed"]) == {"allowed": [1.0, 0.0]}
        assert [n.node_id for n in await store.list_by_user("local")] == ["allowed"]
        hits = await store.search([1.0, 0.0], user_id="local", agent_ids=["agent"])
        assert len(hits) == 1 and hits[0]["node_id"] == "allowed"
        assert set(hits[0]) == {"node_id", "node", "score"}
        hits[0]["node"].content = "caller mutation"
        assert (await store.get_by_id("allowed")).content == allowed.content

    asyncio.run(scenario())
    assert backend.nodes["allowed"].content == allowed.content


@pytest.mark.parametrize(
    "kwargs",
    [{"user_id": "other"}, {"agent_ids": ["other"]}, {"isolation_key": "other::agent::session"}],
)
def test_search_cannot_override_the_bound_namespace(kwargs):
    backend = Backend()
    with pytest.raises(HyMemoryBridgeError, match="scope"):
        asyncio.run(wrapper(backend).search([1.0, 0.0], **kwargs))
    assert backend.calls == []


def test_exact_proof_still_requires_approved_evidence_and_known_content_match():
    allowed = node("known")
    with pytest.raises(HyMemoryBridgeError, match="provenance"):
        HyPermissionVectorStore(
            Backend(allowed),
            Lease(),
            agent_id="agent",
            session_id="session",
            node_permissions={"known": permission(allowed, ("not-approved",))},
        )
    known = HySupportedMemory(
        hy_memory_id="known",
        governed_memory_id="governed",
        revision=1,
        content="Different confirmed content",
        owner="user",
        evidence_ids=("current",),
    )
    store = wrapper(Backend(allowed), lease=Lease((known,)), approved=(allowed,))
    with pytest.raises(HyMemoryBridgeError, match="node_changed"):
        asyncio.run(store.get_by_id("known"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("tags", ["changed"]),
        ("custom", {"other": "secret"}),
        ("embedding", [0.0, 1.0]),
        ("supersedes", ["foreign"]),
    ],
)
def test_full_fingerprint_detects_changes_not_just_content(field, value):
    allowed = node("allowed")
    backend = Backend(allowed)
    store = wrapper(backend, approved=(allowed,))
    setattr(backend.nodes["allowed"], field, value)
    with pytest.raises(HyMemoryBridgeError, match="node_changed"):
        asyncio.run(store.get_by_id("allowed"))


def test_concurrent_authorized_write_can_refresh_a_stale_full_proof():
    allowed = node("allowed")
    backend = Backend(allowed)
    refreshed: list[str] = []

    async def refresh(node_id: str):
        refreshed.append(node_id)
        return permission(backend.nodes[node_id])

    store = wrapper(backend, approved=(allowed,), refresh_permission=refresh)
    backend.nodes["allowed"].embedding = [0.0, 1.0]

    assert asyncio.run(store.get_by_id("allowed")) is not None
    assert refreshed == ["allowed"]


def test_refresh_cannot_authorize_a_node_that_was_not_in_the_original_lease():
    allowed = node("allowed")
    foreign = node("foreign")
    backend = Backend(allowed, foreign)

    async def refresh(_node_id: str):
        return permission(foreign, ("other-evidence",))

    store = wrapper(backend, approved=(allowed,), refresh_permission=refresh)

    assert asyncio.run(store.get_by_id("foreign")) is None


@pytest.mark.parametrize(
    "field", ["supersedes", "superseded_by", "evidence_chain", "source_raw_memory_id"]
)
def test_full_proof_does_not_expand_permission_along_foreign_links(field):
    value = "foreign" if field == "source_raw_memory_id" else ["foreign"]
    head = node("head", **{field: value})
    backend = Backend(head, node("foreign", user_id="other"))
    store = wrapper(backend, approved=(head,))
    assert asyncio.run(store.get_by_id("head")) is None
    assert ("get", "foreign") not in backend.calls


def test_approved_cyclic_evolution_chain_uses_native_algorithm_without_broadening():
    old = node("old", is_latest=False, status=MemoryStatus.SUPERSEDED, superseded_by=["head"])
    head = node("head", supersedes=["old"])
    store = wrapper(Backend(old, head), approved=(old, head))

    async def scenario():
        start = await store.get_by_id("head")
        chain = await _trace_full_chain(store, start)
        assert {n.node_id for n in chain} == {"old", "head"}

    asyncio.run(scenario())


def test_new_raw_write_is_verified_before_exposure_and_tracks_full_lease_footprint():
    raw = node("new-raw", owner=None, layer=MemoryLayer.L1_RAW, content="[user]: Synthetic input")
    backend = Backend()
    lease = Lease()
    store = wrapper(backend, lease=lease)

    async def scenario():
        assert await store.get_by_id(raw.node_id) is None
        assert await store.upsert(raw) == raw.node_id
        assert (await store.get_by_id(raw.node_id)).content == raw.content
        proof = store.written_nodes[raw.node_id]
        assert proof.evidence_ids == tuple(r.evidence_id for r in lease.records)
        assert proof.fingerprint == hy_node_fingerprint(backend.nodes[raw.node_id])
        with pytest.raises(TypeError):
            store.written_nodes["forged"] = proof
        raw.status = MemoryStatus.SHADOW
        assert await store.upsert(raw) == raw.node_id
        assert (await store.get_by_id(raw.node_id)).status is MemoryStatus.SHADOW

    asyncio.run(scenario())


def test_native_history_read_exposes_only_registered_raw_ids():
    raw = node("raw", layer=MemoryLayer.L1_RAW, owner=None, content="[user]: APPROVED_RAW")
    denied = node("denied", layer=MemoryLayer.L1_RAW, owner=None, content="[user]: DENIED_RAW")
    backend = Backend(denied)
    store = wrapper(backend)

    async def scenario():
        await store.upsert(raw)
        history = await MemoryWriter._get_recent_history(None, store, "local", "agent")
        assert "APPROVED_RAW" in history and "DENIED_RAW" not in history

    asyncio.run(scenario())


def test_native_chroma_nullable_chain_roundtrip_preserves_exact_readback_proof():
    class SerializedBackend(Backend):
        async def upsert(self, value):
            result = await super().upsert(value)
            payload = ChromaVectorStore._node_to_payload(value)
            metadata = ChromaVectorStore._payload_to_chroma_metadata(payload)
            restored = ChromaVectorStore._chroma_metadata_to_payload(metadata)
            stored = ChromaVectorStore._payload_to_node(restored)
            stored.embedding = deepcopy(value.embedding)
            self.nodes[result] = stored
            return result

    backend = SerializedBackend()
    store = wrapper(backend)
    original = node("chroma-roundtrip", owner="agent")

    async def scenario():
        assert await store.upsert(original) == original.node_id
        actual = await store.get_by_id(original.node_id)
        assert actual.supersedes == actual.superseded_by == ""
        assert original.supersedes is original.superseded_by is None
        assert actual.content == original.content and actual.owner == "agent"
        assert hy_node_fingerprint(actual) != hy_node_fingerprint(original)
        assert store.written_nodes[original.node_id].fingerprint == hy_node_fingerprint(actual)

    asyncio.run(scenario())


def test_real_cosine_chroma_accepts_dense_vector_roundtrip_without_loosening_node_proof(tmp_path):
    from one_memory.hy_memory_bridge import build_hy_memory_config
    from tests.test_memory_language import _profile

    config = build_hy_memory_config(
        _profile().model_copy(update={"embedding_dimensions": 512}),
        api_key=None,
        data_dir=tmp_path,
    )
    native = ChromaVectorStore(config)
    values = [math.sin(index + 1) / (index + 1) for index in range(512)]
    norm = math.sqrt(sum(value * value for value in values))
    original = node("dense-roundtrip", owner="user", embedding=[value / norm for value in values])

    async def scenario():
        await native.initialize()
        store = wrapper(native)
        try:
            assert await store.upsert(original) == original.node_id
            actual = await store.get_by_id(original.node_id)
            assert actual.content == original.content and actual.owner == original.owner
            assert store.written_nodes[actual.node_id].fingerprint == hy_node_fingerprint(actual)
            changed = deepcopy(actual)
            changed.embedding[20] += 0.001
            await native.upsert(changed)
            with pytest.raises(HyMemoryBridgeError, match="hy_store"):
                await store.get_by_id(original.node_id)
        finally:
            await store.close()
            await native.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["upsert", "update_payload", "update_embedding", "delete"])
def test_guessed_existing_id_is_never_authorized_as_a_new_write(mutation):
    foreign = node("guess", content="DENIED_EXISTING")
    backend = Backend(foreign)
    store = wrapper(backend)

    async def scenario():
        with pytest.raises(HyMemoryBridgeError, match="target_denied"):
            if mutation == "upsert":
                await store.upsert(node("guess", content="replacement"))
            elif mutation == "update_payload":
                await store.update_payload("guess", {"content": "replacement"})
            elif mutation == "update_embedding":
                await store.update_embedding("guess", [0.0, 1.0])
            else:
                await store.delete("guess")

    asyncio.run(scenario())
    assert backend.nodes["guess"].content == foreign.content
    assert all(call[0] == "get" for call in backend.calls)


@pytest.mark.parametrize(
    "changes",
    [
        {"user_id": "other"},
        {"agent_id": "other"},
        {"session_id": "other"},
        {"layer": MemoryLayer.L6_SCHEMA},
        {"supersedes": ["guessed"]},
    ],
)
def test_new_node_cannot_claim_other_scope_or_unapproved_chain(changes):
    backend = Backend()
    with pytest.raises(HyMemoryBridgeError):
        asyncio.run(wrapper(backend).upsert(node("new", **changes)))
    assert backend.nodes == {}


def test_batch_checks_all_targets_before_any_native_write():
    denied = node("existing")
    backend = Backend(denied)
    with pytest.raises(HyMemoryBridgeError, match="target_denied"):
        asyncio.run(wrapper(backend).upsert_batch([node("new"), node("existing")]))
    assert set(backend.nodes) == {"existing"}


def test_approved_native_update_preserves_opaque_metadata_and_refreshes_fingerprint():
    allowed = node("allowed", custom={"opaque": {"source": "approved"}}, tags=["before"])
    backend = Backend(allowed)
    store = wrapper(backend, approved=(allowed,))

    async def scenario():
        assert await store.update_payload("allowed", {"content": "revised", "tags": ["after"]})
        assert await store.update_embedding("allowed", [0.0, 1.0])
        result = await store.get_by_id("allowed")
        assert result.content == "revised" and result.tags == ["after"]
        assert result.custom == allowed.custom and result.embedding == [0.0, 1.0]
        assert store.written_nodes["allowed"].fingerprint == hy_node_fingerprint(result)

    asyncio.run(scenario())


def test_update_cannot_retarget_identity_or_add_denied_chain():
    for updates in (
        {"user_id": "other"},
        {"node_id": "other"},
        {"supersedes": ["outside"]},
        # `custom` remains an allowed field name for SDK 2.0.2; the field it
        # must not smuggle through is the digest-only System 2 counter.
        {"custom": {"s2_evidence_count": 1}},
        {"tag_ners": "not-a-list"},
    ):
        allowed = node("allowed")
        backend = Backend(allowed)
        with pytest.raises(HyMemoryBridgeError):
            asyncio.run(wrapper(backend, approved=(allowed,)).update_payload("allowed", updates))
        assert backend.nodes["allowed"].to_dict() == allowed.to_dict()


def test_native_dedup_can_delete_approved_target_but_not_guessed_cascade():
    async def scenario(approved):
        keep, remove = node("keep"), node("remove")
        backend = Backend(keep, remove)
        store = wrapper(backend, approved=(keep, remove) if approved else (keep,))
        await execute_dedup(
            [
                DedupItem("keep", [1.0, 0.0], gmt_created=1),
                DedupItem("remove", [1.0, 0.0], gmt_created=2),
            ],
            vector_store=store,
            trigger="reconcile",
            threshold=0.95,
        )
        if approved:
            assert backend.nodes["remove"].status is MemoryStatus.SHADOW
            assert backend.nodes["remove"].is_latest is False
            assert backend.nodes["remove"].custom["hy_dedup"] == {
                "reason": "dedup:reconcile",
                "kept_id": "keep",
            }
            assert any(call[:2] == ("update", "remove") for call in backend.calls)
        else:
            assert "remove" in backend.nodes
            with pytest.raises(HyMemoryBridgeError, match="target_denied"):
                store.recheck()  # Native execute_dedup swallowed the failure.

    asyncio.run(scenario(True))
    asyncio.run(scenario(False))


def test_native_dedup_shadows_head_without_destroying_its_history():
    keep = node("keep")
    head = node("remove-head", supersedes=["old"])
    old = node("old", superseded_by=[head.node_id], is_latest=False)
    backend = Backend(keep, head, old)
    store = wrapper(backend, approved=(keep, head, old))

    async def scenario():
        await execute_dedup(
            [
                DedupItem("keep", [1.0, 0.0], gmt_created=1, is_chain_head=True),
                DedupItem(
                    head.node_id,
                    [1.0, 0.0],
                    gmt_created=2,
                    is_chain_head=True,
                    chain_node_ids=[head.node_id, "old"],
                ),
            ],
            vector_store=store,
            trigger="reconcile",
            threshold=0.95,
        )
        store.recheck()
        assert set(backend.nodes) == {"keep", "remove-head", "old"}
        assert backend.nodes["remove-head"].status is MemoryStatus.SHADOW
        assert backend.nodes["remove-head"].is_latest is False
        assert backend.nodes["old"].status is MemoryStatus.SHADOW
        assert backend.nodes["old"].is_latest is False

    asyncio.run(scenario())


def test_forged_dedup_cascade_never_deletes_a_foreign_member():
    keep, head, foreign = node("keep"), node("head"), node("foreign", user_id="other")
    backend = Backend(keep, head, foreign)
    store = wrapper(backend, approved=(keep, head))

    async def scenario():
        await execute_dedup(
            [
                DedupItem("keep", [1.0, 0.0], gmt_created=1, is_chain_head=True),
                DedupItem(
                    "head",
                    [1.0, 0.0],
                    gmt_created=2,
                    is_chain_head=True,
                    chain_node_ids=["head", "foreign"],
                ),
            ],
            vector_store=store,
            trigger="reconcile",
            threshold=0.95,
        )
        assert "foreign" in backend.nodes
        with pytest.raises(HyMemoryBridgeError, match="target_denied"):
            store.recheck()

    asyncio.run(scenario())


def test_basic_profile_native_write_and_supersede_stay_on_same_backend():
    old = node(
        "old-profile",
        owner=None,
        layer=MemoryLayer.L0_BASIC_INFO,
        content="name: Before",
        custom={"basic_info_kv": {"name": "Before"}},
    )
    backend = Backend(old)
    store = wrapper(backend, approved=(old,))

    class Embedder:
        async def embed_queued(self, text):
            return [1.0, 0.0]

    async def scenario():
        result = await upsert_basic_profile(
            user_id="local",
            kv={"name": "After"},
            vector_store=store,
            embed_service=Embedder(),
            allowed_fields=["name"],
        )
        assert result.success and result.changed
        assert set(backend.nodes) == {"old-profile", result.node_id}
        assert (await store.get_by_id(result.node_id)).supersedes == ["old-profile"]
        assert backend.nodes["old-profile"].superseded_by == [result.node_id]

    asyncio.run(scenario())


def test_open05_can_migrate_an_approved_legacy_agent_profile_to_user_level():
    old = node(
        "old-profile",
        agent_id="legacy-agent",
        owner=None,
        layer=MemoryLayer.L0_BASIC_INFO,
        content="name: Before",
        custom={"basic_info_kv": {"name": "Before"}},
    )
    backend = Backend(old)
    store = wrapper(backend, approved=(old,))

    class Embedder:
        async def embed_queued(self, text):
            return [1.0, 0.0]

    async def scenario():
        result = await upsert_basic_profile(
            user_id="local",
            kv={"name": "After"},
            vector_store=store,
            embed_service=Embedder(),
            allowed_fields=["name"],
        )
        assert result.success and result.changed
        migrated = await store.get_by_id(result.node_id)
        assert migrated is not None
        assert migrated.agent_id == "" and migrated.session_id == ""
        assert migrated.supersedes == ["old-profile"]

    asyncio.run(scenario())


def test_revocation_during_read_cannot_release_payload_or_perform_write():
    for mutation in (False, True):
        allowed = node("allowed")
        backend, lease = Backend(allowed), Lease()
        store = wrapper(backend, lease=lease, approved=(allowed,))
        backend.after_read = lambda lease=lease: setattr(lease, "revoked", True)
        with pytest.raises(HyMemoryBridgeError, match="permission_revoked"):
            asyncio.run(
                store.update_payload("allowed", {"content": "changed"})
                if mutation
                else store.get_by_id("allowed")
            )
        assert backend.nodes["allowed"].content == allowed.content


def test_multi_node_read_rechecks_permission_at_public_boundary_not_per_node():
    nodes = tuple(node(f"allowed-{index}") for index in range(8))
    backend, lease = Backend(*nodes), Lease()
    store = wrapper(backend, lease=lease, approved=nodes)
    checks_after_construction = lease.checks

    result = asyncio.run(store.list_by_user("local"))

    assert {item.node_id for item in result} == {item.node_id for item in nodes}
    assert lease.checks - checks_after_construction == 2


def test_corrupt_native_write_is_not_registered_or_exposed():
    backend = Backend()
    backend.corrupt_write = lambda n: setattr(n, "custom", {"unknown": "DENIED"})
    store = wrapper(backend)
    with pytest.raises(HyMemoryBridgeError, match="write_mismatch"):
        asyncio.run(store.upsert(node("new")))
    with pytest.raises(HyMemoryBridgeError, match="write_mismatch"):
        asyncio.run(store.get_by_id("new"))


def test_successful_batch_and_deleted_ids_cannot_be_recreated_under_new_permission():
    backend = Backend()
    store = wrapper(backend)

    async def scenario():
        assert await store.upsert_batch([node("first"), node("second")]) == ["first", "second"]
        assert set(store.written_nodes) == {"first", "second"}
        await store.delete("first")
        with pytest.raises(HyMemoryBridgeError):
            await store.upsert(node("first"))
        assert set(backend.nodes) == {"second"}

    asyncio.run(scenario())


def test_invalid_node_fingerprint_latches_even_if_native_caller_swallows_exception():
    backend = Backend()
    store = wrapper(backend)
    invalid = node("invalid", custom={"unsupported_object": object()})
    with pytest.raises(HyMemoryBridgeError, match="node_shape"):
        asyncio.run(store.upsert(invalid))
    with pytest.raises(HyMemoryBridgeError, match="node_shape"):
        store.recheck()
    assert backend.nodes == {}


@pytest.mark.parametrize(
    ("expected_tags", "stored_tags"),
    [([], "[]"), (["release", "permission"], '["release", "permission"]')],
)
def test_public_sdk_tag_ners_chroma_roundtrip_is_allowed(expected_tags, stored_tags):
    expected = node("new")
    expected.tag_ners = expected_tags
    backend = Backend()
    backend.corrupt_write = lambda n: setattr(n, "tag_ners", stored_tags)
    store = wrapper(backend)

    assert asyncio.run(store.upsert(expected)) == "new"
    assert set(store.written_nodes) == {"new"}


def test_malformed_tag_ners_remains_a_write_mismatch():
    expected = node("new")
    expected.tag_ners = ["release"]
    backend = Backend()
    backend.corrupt_write = lambda n: setattr(n, "tag_ners", "not-json")
    store = wrapper(backend)

    with pytest.raises(HyMemoryBridgeError, match="write_mismatch"):
        asyncio.run(store.upsert(expected))


def test_public_sdk_list_options_stay_inside_the_leased_user_and_agent():
    allowed = node("allowed")
    backend = Backend(allowed)
    store = wrapper(backend, approved=(allowed,))

    rows = asyncio.run(
        store.list_by_user(
            user_id="local",
            agent_id="agent",
            user_ids=["local"],
            agent_ids=["agent"],
            only_latest=True,
            with_vectors=False,
            allow_raw_layer=True,
        )
    )

    assert [value.node_id for value in rows] == ["allowed"]
    assert next(call for call in backend.calls if call[0] == "list") == (
        "list",
        {
            "user_id": "local",
            "agent_id": "agent",
            "limit": 10000,
            "status_filter": None,
            "layers": None,
            "user_ids": ["local"],
            "agent_ids": ["agent"],
            "only_latest": True,
            "with_vectors": False,
            "allow_raw_layer": True,
        },
    )

    with pytest.raises(HyMemoryBridgeError, match="scope"):
        asyncio.run(store.list_by_user(user_ids=["other-user"]))


def test_two_wrappers_never_share_new_node_permissions():
    backend = Backend()
    first, second = wrapper(backend), wrapper(backend)

    async def scenario():
        await first.upsert(node("first-only"))
        assert await second.get_by_id("first-only") is None
        assert (await first.get_by_id("first-only")).node_id == "first-only"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "action",
    [
        "id",
        "ids",
        "embeddings",
        "list",
        "search",
        "upsert",
        "batch",
        "payload",
        "embedding",
        "delete",
        "initialize",
    ],
)
def test_revocation_is_checked_before_every_native_action_even_empty_reads(action):
    approved = node("approved")
    backend, lease = Backend(approved), Lease()
    store = wrapper(backend, lease=lease, approved=(approved,))
    lease.revoked = True

    async def scenario():
        calls = {
            "id": lambda: store.get_by_id("approved"),
            "ids": lambda: store.get_by_ids([]),
            "embeddings": lambda: store.get_embeddings([]),
            "list": lambda: store.list_by_user("local"),
            "search": lambda: store.search([1.0, 0.0]),
            "upsert": lambda: store.upsert(node("new")),
            "batch": lambda: store.upsert_batch([]),
            "payload": lambda: store.update_payload("approved", {}),
            "embedding": lambda: store.update_embedding("approved", [0.0, 1.0]),
            "delete": lambda: store.delete("approved"),
            "initialize": store.initialize,
        }
        with pytest.raises(HyMemoryBridgeError, match="permission_revoked"):
            await calls[action]()
        lease.revoked = False
        with pytest.raises(HyMemoryBridgeError, match="permission_revoked"):
            store.recheck()

    asyncio.run(scenario())
    assert backend.calls == []


def test_identical_text_and_vector_never_authorize_a_different_native_id():
    allowed = node("allowed")
    denied = node("denied", content=allowed.content, embedding=allowed.embedding)
    backend = Backend(allowed, denied)
    store = wrapper(backend, approved=(allowed,))

    async def scenario():
        assert [hit["node_id"] for hit in await store.search([1.0, 0.0])] == ["allowed"]
        assert await store.get_by_id("denied") is None

    asyncio.run(scenario())


def test_invalid_embedding_is_rejected_before_native_mutation():
    for mutation in ("payload", "embedding"):
        approved = node("approved")
        backend = Backend(approved)
        store = wrapper(backend, approved=(approved,))
        with pytest.raises(HyMemoryBridgeError, match="node_shape"):
            asyncio.run(
                store.update_payload("approved", {"embedding": [1e200]})
                if mutation == "payload"
                else store.update_embedding("approved", [1e200])
            )
        assert not any(call[0] in {"update", "embedding"} for call in backend.calls)


def test_distinct_permission_wrappers_serialize_native_store_mutations():
    class ContendedBackend(Backend):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.peak = 0

        async def upsert(self, value):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.02)
                return await super().upsert(value)
            finally:
                self.active -= 1

    async def scenario():
        backend = ContendedBackend()
        first = wrapper(backend)
        second = wrapper(backend)
        await asyncio.gather(first.upsert(node("first")), second.upsert(node("second")))
        assert backend.peak == 1
        assert set(backend.nodes) == {"first", "second"}

    asyncio.run(scenario())


@pytest.mark.parametrize("initial", ["new", "preexisting"])
@pytest.mark.parametrize("mutation", ["upsert", "upsert_batch", "payload", "embedding"])
def test_latest_write_proof_replaces_prior_snapshot_for_every_mutation(initial, mutation):
    original = node("changing")
    backend = Backend(original) if initial == "preexisting" else Backend()
    store = wrapper(backend, approved=(original,) if initial == "preexisting" else ())

    async def scenario():
        if initial == "new":
            await store.upsert(original)
        else:
            await store.update_payload(original.node_id, {"tags": ["first-update"]})
        before = store.written_nodes
        prior_fingerprint = before[original.node_id].fingerprint
        changed = await store.get_by_id(original.node_id)
        changed.content = "Synthetic revised content"
        if mutation == "upsert":
            await store.upsert(changed)
        elif mutation == "upsert_batch":
            await store.upsert_batch([changed])
        elif mutation == "payload":
            await store.update_payload(original.node_id, {"content": changed.content})
        else:
            await store.update_embedding(original.node_id, [0.0, 1.0])

        latest = store.written_nodes[original.node_id]
        assert latest.fingerprint != prior_fingerprint
        assert latest.fingerprint == hy_node_fingerprint(backend.nodes[original.node_id])
        assert latest.evidence_ids == ("current", "support")
        assert before[original.node_id].fingerprint == prior_fingerprint
        assert hy_node_fingerprint(await store.get_by_id(original.node_id)) == latest.fingerprint

    asyncio.run(scenario())


def test_embedding_write_accepts_native_modified_time_and_refreshes_full_proof():
    original = node("modified-at")
    backend = Backend(original)

    async def update_embedding(key, embedding):
        backend.nodes[key].embedding = deepcopy(embedding)
        backend.nodes[key].gmt_modified += timedelta(seconds=1)
        return True

    backend.update_embedding = update_embedding
    store = wrapper(backend, approved=(original,))

    async def scenario():
        await store.update_embedding(original.node_id, [0.0, 1.0])
        assert store.written_nodes[original.node_id].fingerprint == hy_node_fingerprint(
            backend.nodes[original.node_id]
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("initial", ["new", "preexisting"])
def test_deletion_delta_removes_persisted_proof_without_replacing_unmodified_rows(initial):
    removed, untouched = node("removed"), node("untouched")
    backend = Backend(removed, untouched) if initial == "preexisting" else Backend(untouched)
    supplied = {untouched.node_id: permission(untouched)}
    if initial == "preexisting":
        supplied[removed.node_id] = permission(removed)
    store = HyPermissionVectorStore(
        backend,
        Lease(),
        agent_id="agent",
        session_id="session",
        node_permissions=supplied,
    )

    async def scenario():
        if initial == "new":
            await store.upsert(removed)
        previous_deleted = store.deleted_node_ids
        ledger = {**supplied, **store.written_nodes}
        await store.delete(removed.node_id)
        assert store.deleted_node_ids == frozenset({removed.node_id})
        assert previous_deleted == frozenset()
        assert removed.node_id not in store.written_nodes
        assert await store.get_by_id(removed.node_id) is None

        ledger.update(store.written_nodes)
        for key in store.deleted_node_ids:
            ledger.pop(key, None)
        assert set(ledger) == {untouched.node_id}
        await store.close()
        reopened = HyPermissionVectorStore(
            backend,
            Lease(),
            agent_id="agent",
            session_id="session",
            node_permissions=ledger,
        )
        assert (await reopened.get_by_id(untouched.node_id)).node_id == untouched.node_id
        assert await reopened.get_by_id(removed.node_id) is None
        await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("snapshot", ["written_nodes", "deleted_node_ids"])
def test_proof_snapshots_are_permission_checked_before_export(snapshot):
    lease = Lease()
    store = wrapper(Backend(), lease=lease)
    lease.revoked = True
    with pytest.raises(HyMemoryBridgeError, match="permission_revoked"):
        getattr(store, snapshot)


def test_close_after_parent_commit_never_rechecks_invalidated_context_or_backend():
    class CommitSensitiveLease(Lease):
        committed = False

        def recheck(self):
            if self.committed:
                self.checks += 1
                raise HyMemoryBridgeError("model_content_context_changed")
            return super().recheck()

    backend, lease = Backend(), CommitSensitiveLease()
    store = wrapper(backend, lease=lease)

    async def scenario():
        await store.upsert(node("new"))
        snapshot = store.written_nodes
        deleted = store.deleted_node_ids
        lease.committed = True
        checks_before_close, calls_before_close = lease.checks, list(backend.calls)
        await store.close()
        await store.close()
        assert lease.checks == checks_before_close
        assert backend.calls == calls_before_close
        assert set(snapshot) == {"new"} and not deleted
        with pytest.raises(HyMemoryBridgeError, match="closed"):
            await store.get_by_id("new")
        assert lease.checks == checks_before_close

    asyncio.run(scenario())


def test_wrapper_api_is_closed_and_close_does_not_close_or_modify_native_store():
    backend, lease = Backend(node("kept")), Lease()
    store = wrapper(backend, lease=lease)
    for attribute in (
        "_collection",
        "_client",
        "native",
        "backend",
        "_run",
        "get_node",
        "upsert_entity",
        "search_entities",
        "list_entities",
        "sweep",
        "digest",
        "delete_by_metadata",
        "delete_by_isolation_key",
    ):
        assert not hasattr(store, attribute), attribute

    async def scenario():
        await store.initialize()
        with pytest.raises(NotImplementedError):
            await store.has_tag_embedding("local", "tag")
        await store.close()
        with pytest.raises(HyMemoryBridgeError, match="closed"):
            await store.get_by_id("kept")

    asyncio.run(scenario())
    assert set(backend.nodes) == {"kept"} and backend.calls == []


def test_allowing_owner_and_custom_does_not_reach_another_users_node():
    """The pinned public SDK's reconcile UPDATE carries `owner` / `custom`.

    Allowing those two field names is only safe while the store still resolves
    the target through the lease's own permissions and user scope. A node that
    belongs to a different user_id must remain unreachable, so the update is
    refused as a denied target rather than being applied.
    """
    foreign = node("foreign", user_id="someone-else")
    backend = Backend(foreign)
    store = wrapper(backend, approved=(foreign,))

    async def scenario():
        assert await store.get_by_id(foreign.node_id) is None
        with pytest.raises(HyMemoryBridgeError, match="target_denied"):
            await store.update_payload(foreign.node_id, {"owner": "agent", "custom": {}})
        with pytest.raises(HyMemoryBridgeError, match="target_denied"):
            await store.update_payload(foreign.node_id, {"custom": {"note": "DENIED"}})

    asyncio.run(scenario())
    assert backend.nodes[foreign.node_id] == foreign
    assert not any(call[0] == "update" for call in backend.calls)


def test_owner_and_custom_updates_are_allowed_for_an_approved_own_node():
    """The two public SDK fields apply when the node is this lease's own."""
    owned = node("owned", custom={})
    backend = Backend(owned)
    store = wrapper(backend, approved=(owned,))

    async def scenario():
        assert await store.update_payload(owned.node_id, {"owner": "agent", "custom": {"n": 1}})

    asyncio.run(scenario())
    assert backend.nodes[owned.node_id].owner == "agent"
    assert backend.nodes[owned.node_id].custom == {"n": 1}
