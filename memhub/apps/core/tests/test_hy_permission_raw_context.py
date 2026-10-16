"""Real pinned HY SDK/Chroma and guards; scripted HTTP is not model acceptance."""

import asyncio
import json
import sqlite3
import traceback
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from hy_memory.client import HyMemoryClient
from hy_memory.data.vector_store_chroma import ChromaVectorStore
from hy_memory.models.memory import MemoryLayer, MemoryNode
from hy_memory.pipelines._retrieval.reconcile_retrieval import ReconcileHybridRetriever

from one_memory.hy_permission_context import permission_bridge_factory
from one_memory.hy_response_guard import HyReconcileGuard, HyResponseGuard
from one_memory.hy_runtime_config import hy_memory_version
from one_memory.hy_write_verification import HyExtractionGuard
from one_memory.imports.grants import SourceGrantStore
from one_memory.imports.models import (
    EvidenceContentKind,
    EvidenceInput,
    EvidenceOrigin,
    OriginMetadata,
    WrittenBy,
)
from one_memory.imports.sources.registry import RootCandidateKind
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_governance import MemoryGovernanceStore
from one_memory.memory_processing import MemoryProcessingSupervisor
from one_memory.memory_tasks import MemoryTaskStore, evidence_ref
from one_memory.model_content_permissions import ModelContentPermissionStore
from one_memory.model_profiles import (
    ModelCapabilities,
    ModelProfileInput,
    ModelProfileStore,
    ProviderKind,
)
from one_memory.secrets import InMemorySecretStore

USER_TEXT = "The user requires a permission review before the synthetic release."
AGENT_TEXT = "The agent proposes a checklist for the synthetic release."
CLAIMS = {USER_TEXT: "user", AGENT_TEXT: "agent"}
DENIED_FACT = "DENIED_NATIVE_FACT_63a1: omit the permission review."
DENIED_HISTORY = "DENIED_NATIVE_HISTORY_86b2: publish without asking."
DENIED_TAG = "DENIED_NATIVE_TAG_91c3"
DENIED_IDS = ("denied-native-fact", "denied-native-history")
NOW = datetime(2026, 9, 13, 8, tzinfo=UTC)


def vector(text):
    # Orthogonal synthetic vectors keep the two claims distinct under native dedup.
    axis = 1 if text == USER_TEXT else 2 if text == AGENT_TEXT else 0
    return [float(index == axis) for index in range(384)]


def prompt_json(prompt, heading):
    payload = prompt_section(prompt, heading)
    starts = [index for marker in ("[", "{") if (index := payload.find(marker)) >= 0]
    assert starts, (heading, payload)
    payload = payload[min(starts) :]
    try:
        return json.JSONDecoder().raw_decode(payload)[0]
    except json.JSONDecodeError as error:
        raise AssertionError(f"expected JSON after {heading!r}, got {payload[:500]!r}") from error


def prompt_section(prompt: str, heading: str) -> str:
    assert prompt.count(heading) == 1, heading
    section = prompt.split(heading, 1)[1]
    return section.split("\n---\n", 1)[0].strip()


class ScriptedOracle:
    """Only the two fixture claims receive support, with exact supplied Evidence IDs."""

    def __init__(self, records, *, after_response=None):
        self.records = {record.evidence_id: record for record in records}
        self.requests = []
        self.stages = []
        self.reconciles = []
        self.errors = []
        self.after_response = after_response
        self.revoked_at = None
        self.limit = 24

    def __call__(self, request):
        try:
            return self.respond(request)
        except Exception:
            self.errors.append(traceback.format_exc())
            raise

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        assert len(self.requests) <= self.limit, "scripted HTTP response budget exceeded"
        assert self.revoked_at is None, "native child request escaped after revocation"
        assert request.method == "POST"
        serialized = json.dumps(body)
        for forbidden in (DENIED_FACT, DENIED_HISTORY, DENIED_TAG, *DENIED_IDS):
            assert forbidden not in serialized, f"denied native context reached HTTP: {forbidden}"
        if request.url.path == "/v1/embeddings":
            assert str(request.url) == "http://127.0.0.1:2/v1/embeddings"
            assert body["model"] == "raw-fixture-embedding"
            texts = body["input"]
            assert isinstance(texts, list) and texts
            lines = [f"[{record.role}]: {record.verbatim}" for record in self.records.values()]
            # HY stores one raw copy per add, and each add is one turn.
            turns = {
                "\n".join(lines[start:end])
                for start in range(len(lines))
                for end in range(start + 1, len(lines) + 1)
            }
            for text in texts:
                if text not in CLAIMS:
                    assert text in turns
            self.stages.append("embedding")
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {"index": i, "object": "embedding", "embedding": vector(text)}
                        for i, text in enumerate(texts)
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        assert str(request.url) == "http://127.0.0.1:1/v1/chat/completions"
        assert body["model"] == "raw-fixture-generation"
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("You are a single-label scene classifier."):
            result = {"is_coding": False, "reason": "synthetic chat"}
            stage = "judge"
        elif "## New Messages\n" in prompt:
            self.check_turns(prompt_json(prompt, "## New Messages\n"))
            history_section = prompt_section(prompt, "## Last k Messages\n")
            history = (
                [] if history_section == "(none)" else prompt_json(prompt, "## Last k Messages\n")
            )
            for source in history:
                record = self.records[source["evidence_id"]]
                assert source["content"] == record.verbatim
                assert source["role"] == record.role
                assert source["revision"] == record.revision
            assert '"memory"' in prompt
            assert ('"intentions"' in prompt) is ("never" not in prompt)
            result = {
                "memory": [
                    {"content": text, "owner": owner, "tags": []} for text, owner in CLAIMS.items()
                ],
                "basic_info": {},
            }
            if '"intentions"' in prompt:
                result["intentions"] = []
            stage = "extract"
        elif "## Existing memories\n" in prompt:
            existing = prompt_json(prompt, "## Existing memories\n")
            incoming = prompt_json(prompt, "## New memories\n")
            assert existing and incoming
            for item in (*existing, *incoming):
                assert CLAIMS[item["content"]] == item["owner"]
                assert not item.get("tags")
            self.reconciles.append(existing)
            by_content = {item["content"]: item for item in existing}
            result = [
                {
                    "op": "UPDATE" if item["content"] in by_content else "ADD",
                    "content": item["content"],
                    "owner": item["owner"],
                    "layer": "L2_FACT",
                    "tags": [],
                    **(
                        {"memory_id": by_content[item["content"]]["memory_id"]}
                        if item["content"] in by_content
                        else {}
                    ),
                }
                for item in incoming
            ]
            stage = "reconcile"
        else:
            if not prompt.lstrip().startswith("{"):
                raise AssertionError(f"unknown native child prompt: {prompt[:300]!r}")
            data = json.loads(prompt)
            if set(data) == {"items", "sources", "before"}:
                result = self.review(data)
                stage = "review"
            else:
                assert set(data) == {"proofs"}, "unknown native child prompt"
                result = self.cited_review(data)
                stage = "citations"
        self.stages.append(stage)
        response = httpx.Response(
            200,
            json={
                "id": "synthetic-raw-response",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
        if self.after_response is not None and self.after_response(stage):
            self.revoked_at = len(self.requests)
        return response

    def check_turns(self, turns):
        assert turns and isinstance(turns, list)
        allowed = {(record.role, record.verbatim) for record in self.records.values()}
        assert all(set(turn) == {"role", "content"} for turn in turns)
        assert {(turn["role"], turn["content"]) for turn in turns} <= allowed

    def review(self, data):
        assert data["items"] and data["sources"]
        for source in data["sources"]:
            record = self.records[source["id"]]
            assert source["role"] == record.role
            assert source["revision"] == record.revision
            assert source["created_at"] == record.created_at.isoformat()
            assert "".join(span["text"] for span in source["spans"]) == record.verbatim
        reviews = []
        for item in data["items"]:
            assert CLAIMS[item["content"]] == item["owner"]
            role = "user" if item["owner"] == "user" else "assistant"
            source = next(
                source
                for source in data["sources"]
                if source["role"] == role and self.records[source["id"]].verbatim == item["content"]
            )
            reviews.append(
                {
                    "id": item["id"],
                    "clauses": [
                        {
                            "text": item["content"],
                            "verdict": "supported",
                            "authority": item["owner"],
                            "citations": [
                                {
                                    "source_id": source["id"],
                                    "span_ids": [span["id"] for span in source["spans"]],
                                }
                            ],
                            "reason": "exact synthetic fixture claim",
                        }
                    ],
                }
            )
        for before in data["before"]:
            assert any(
                (item["content"], item["owner"]) == (before["content"], before["owner"])
                for item in data["items"]
            )
        return {
            "reviews": reviews,
            "coverage": [
                {"id": item["id"], "verdict": "preserved", "reason": "exact fixture match"}
                for item in data["before"]
            ],
        }

    def cited_review(self, data):
        assert data["proofs"]
        for proof in data["proofs"]:
            assert CLAIMS[proof["claim"]] == proof["owner"]
            assert proof["citations"] and not proof["context_citations"]
            for citation in proof["citations"]:
                record = self.records[citation["source_id"]]
                assert citation["text"] == record.verbatim == proof["claim"]
                assert citation["role"] == record.role
                assert citation["revision"] == record.revision
                assert citation["created_at"] == record.created_at.isoformat()
        return {
            "proofs": [
                {"id": proof["id"], "verdict": "supported", "reason": "exact fixture match"}
                for proof in data["proofs"]
            ]
        }


def put_turn(evidence, message_id, role, text, ordinal, *, conversation_id="raw-session"):
    return evidence.put(
        EvidenceInput(
            user_id="local",
            source_id="raw-fixture",
            conversation_id=conversation_id,
            message_id=message_id,
            source_ordinal=str(ordinal),
            role=role,
            written_by={"user": WrittenBy.PERSON, "tool": WrittenBy.TOOL}.get(
                role, WrittenBy.AGENT
            ),
            content_kind=EvidenceContentKind.RAW_CONVERSATION,
            created_at=NOW + timedelta(seconds=ordinal),
            verbatim=text,
            origin_meta={EvidenceOrigin.MANUAL_IMPORT: OriginMetadata()},
        )
    ).record


def grant_records(supervisor, records, expected_revision=0):
    profile = supervisor.profiles.active(supervisor.secrets)
    snapshot = supervisor.model_content_permissions.snapshot(
        user_id="local",
        source_ids=("raw-fixture",),
        profile=supervisor.profiles.snapshot(profile.profile_id, profile.revision),
        evidence=records,
    )
    return supervisor.model_content_permissions.grant(
        snapshot, user_confirmed=True, expected_revision=expected_revision
    )


@asynccontextmanager
async def raw_client(tmp_path, *, tool=None, conversation_id="raw-session"):
    assert hy_memory_version() == "2.0.2"
    root = tmp_path / "source"
    root.mkdir()
    database = tmp_path / "one-memory.sqlite3"
    grants = SourceGrantStore(database, home=tmp_path)
    grants.grant(
        user_id="local",
        source_id="raw-fixture",
        root=root,
        candidate_kind=RootCandidateKind.DEFAULT,
    )
    evidence = EvidenceStore(database)
    records = (
        put_turn(
            evidence,
            "user-original",
            "user",
            USER_TEXT,
            1,
            conversation_id=conversation_id,
        ),
        put_turn(
            evidence,
            "agent-original",
            "assistant",
            AGENT_TEXT,
            2,
            conversation_id=conversation_id,
        ),
        *(
            (
                put_turn(
                    evidence,
                    "tool-original",
                    "tool",
                    tool,
                    3,
                    conversation_id=conversation_id,
                ),
            )
            if tool
            else ()
        ),
    )
    profiles, secrets = ModelProfileStore(database), InMemorySecretStore()
    profile = profiles.create(
        ModelProfileInput(
            name="Synthetic HTTP only",
            provider_kind=ProviderKind.OLLAMA,
            base_url="http://127.0.0.1:1/v1",
            model="raw-fixture-generation",
            embedding_provider_kind=ProviderKind.OLLAMA,
            embedding_base_url="http://127.0.0.1:2/v1",
            embedding_model="raw-fixture-embedding",
            embedding_dimensions=384,
        ),
        secrets,
    )
    profiles.record_capabilities(
        profile.profile_id,
        expected_revision=profile.revision,
        capabilities=ModelCapabilities(
            endpoint_reachable=True, model_available=True, embedding_available=True
        ),
        secrets=secrets,
    )
    profiles.activate(profile.profile_id, secrets)
    supervisor = MemoryProcessingSupervisor(
        tasks=MemoryTaskStore(database),
        profiles=profiles,
        secrets=secrets,
        evidence=evidence,
        governance=MemoryGovernanceStore(database),
        data_dir=tmp_path,
        model_content_permissions=ModelContentPermissionStore(database),
    )
    try:
        permission = grant_records(supervisor, records)
        assert supervisor.governance.list(user_id="local") == []
        yield supervisor, records, permission
    finally:
        await supervisor.shutdown()


def test_search_after_first_wb20_task_does_not_poison_the_second_task(tmp_path):
    async def scenario():
        async with raw_client(tmp_path, conversation_id="wb-20") as (
            supervisor,
            records,
            permission,
        ):
            first = supervisor.prepare_conversation(records[0])[0]
            _, request, _ = supervisor._create_runner()._prepare_permission_lease(first)
            client = await native_client(supervisor, first)
            duplicate = MemoryNode(
                node_id="older-identical-memory",
                user_id=request.user_id,
                agent_id=request.agent_id,
                session_id="wb-19",
                layer=MemoryLayer.L2_FACT,
                content=USER_TEXT,
                owner="user",
                embedding=vector(USER_TEXT),
                gmt_created=NOW - timedelta(days=1),
                gmt_modified=NOW - timedelta(days=1),
            )
            await on_native_loop(client, client._vector_store.upsert(duplicate))

            first_oracle = ScriptedOracle(records)
            bind_oracle(supervisor, first_oracle)
            first_result = await supervisor._create_runner().run(first.task_id)
            assert first_result.status == "completed", first_result.last_error_code

            current = next(
                memory
                for memory in supervisor.governance.list(user_id="local")
                if memory.content == USER_TEXT
            )
            client._spawn_search_dedup(
                [
                    {
                        "memory_id": duplicate.node_id,
                        "content": duplicate.content,
                        "layer": duplicate.layer.value,
                        "gmt_created": int(duplicate.gmt_created.timestamp()),
                    },
                    {
                        "memory_id": current.memory_id,
                        "content": current.content,
                        "layer": MemoryLayer.L2_FACT.value,
                        "gmt_created": int(NOW.timestamp()),
                    },
                ],
                request_id="wb-20-search",
                user_id=request.user_id,
                agent_id=request.agent_id,
            )
            await asyncio.sleep(0.2)

            added = put_turn(
                supervisor.evidence,
                "wb-20-second",
                "user",
                USER_TEXT,
                3,
                conversation_id="wb-20",
            )
            records = (*records, added)
            grant_records(supervisor, records, permission.revision)
            second = supervisor.prepare_conversation(added)[0]
            second_oracle = ScriptedOracle(records)
            bind_oracle(supervisor, second_oracle)
            second_result = await supervisor._create_runner().run(second.task_id)

            assert second_result.status == "completed", second_result.last_error_code
            assert second_result.last_error_code is None

    asyncio.run(scenario())


async def native_client(supervisor, task):
    bridge = supervisor._bridge(task.profile_snapshot, None, supervisor.data_dir)
    client = await bridge._get_client()
    assert isinstance(client, HyMemoryClient)
    assert isinstance(client._vector_store, ChromaVectorStore)
    guards = client._one_memory_write_guards
    assert isinstance(guards[0], HyResponseGuard)
    assert isinstance(guards[2], HyReconcileGuard)
    assert isinstance(guards[3], HyExtractionGuard)
    return client


def test_governance_write_runs_after_shared_hy_bridge_is_released(tmp_path):
    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, _permission):
            task = supervisor.prepare_conversation(records[0])[0]
            bind_oracle(supervisor, ScriptedOracle(records))
            await native_client(supervisor, task)
            bridge = next(
                value
                for key, value in supervisor._bridges.items()
                if key[:2] == (task.profile_id, task.profile_revision)
            )
            runner = supervisor._create_runner()
            original = runner._record_outputs
            observed = []

            def checked(*args, **kwargs):
                observed.append(bridge._access_owner)
                return original(*args, **kwargs)

            runner._record_outputs = checked
            result = await runner.run(task.task_id)

            assert result.status == "completed"
            assert observed == [None], "governance writes must not hold the shared HY bridge"

    asyncio.run(scenario())


async def on_native_loop(client, operation):
    return await asyncio.to_thread(client._loop_thread.run, operation)


async def seed_denied(client, request):
    store = client._vector_store
    nodes = (
        MemoryNode(
            node_id=DENIED_IDS[0],
            user_id=request.user_id,
            agent_id=request.agent_id,
            session_id=request.session_id,
            layer=MemoryLayer.L2_FACT,
            content=DENIED_FACT,
            owner="user",
            embedding=vector(USER_TEXT),
            tags=[DENIED_TAG],
        ),
        MemoryNode(
            node_id=DENIED_IDS[1],
            user_id=request.user_id,
            agent_id=request.agent_id,
            session_id=request.session_id,
            layer=MemoryLayer.L1_RAW,
            content=f"[user]: {DENIED_HISTORY}",
            owner="user",
            embedding=vector("raw"),
            tags=[DENIED_TAG],
        ),
    )
    await on_native_loop(client, store.upsert_batch(list(nodes)))
    retrieved = await on_native_loop(
        client,
        store.search(
            query_embedding=vector(USER_TEXT),
            user_id=request.user_id,
            agent_ids=[request.agent_id],
            layers=[MemoryLayer.L2_FACT],
        ),
    )
    assert {hit["node_id"] for hit in retrieved} == {DENIED_IDS[0]}
    history = await on_native_loop(
        client,
        store.list_by_user(
            user_id=request.user_id,
            agent_id=request.agent_id,
            layers=[MemoryLayer.L1_RAW],
            allow_raw_layer=True,
        ),
    )
    assert [node.node_id for node in history] == [DENIED_IDS[1]]
    candidates = await on_native_loop(
        client,
        ReconcileHybridRetriever().search_candidates(
            USER_TEXT,
            vector(USER_TEXT),
            vector_store=store,
            user_id=request.user_id,
            agent_id=request.agent_id,
            layers=[MemoryLayer.L2_FACT],
        ),
    )
    assert [hit["node"].node_id for hit in candidates] == [DENIED_IDS[0]]
    pipeline = await on_native_loop(client, client._registry.get_system2_writer())
    native_history = await on_native_loop(
        client,
        pipeline._writer._get_recent_history(store, request.user_id, request.agent_id),
    )
    native_tags = await on_native_loop(
        client,
        pipeline._writer._collect_existing_tags(store, request.user_id, request.agent_id),
    )
    assert DENIED_HISTORY in native_history
    assert DENIED_TAG in native_tags
    return await node_snapshots(client, DENIED_IDS)


async def node_snapshots(client, identifiers):
    result = {}
    for identifier in identifiers:
        node = await on_native_loop(client, client._vector_store.get_by_id(identifier))
        assert node is not None, identifier
        result[identifier] = node.to_dict()
    return result


def bind_oracle(supervisor, oracle):
    supervisor.permission_bridge_factory = permission_bridge_factory(
        supervisor._bridge, http_transport=httpx.MockTransport(oracle)
    )


async def assert_outputs(supervisor, task, records, client, *, raw_records=None):
    outputs = supervisor.tasks.outputs(task.task_id)
    assert len(outputs) == 2
    memories = supervisor.governance.list(user_id="local")
    assert {memory.content for memory in memories} == set(CLAIMS)
    by_content = {memory.content: memory for memory in memories}
    for record in records[:2]:
        memory = by_content[record.verbatim]
        output = next(output for output in outputs if output.memory_id == memory.memory_id)
        assert output.owner == CLAIMS[record.verbatim]
        assert output.engine == "hy-memory" and output.engine_version == "2.0.2"
        assert output.hy_memory_id == memory.memory_id
        assert output.memory_revision == memory.revision
        assert output.support_refs == (evidence_ref(record),)
        assert memory.evidence_ids == (record.evidence_id,)
        # Usable either way; origin is what keeps an agent report from passing as the person.
        assert memory.status == "confirmed"
        assert memory.origin == ("agent" if output.owner == "agent" else "extracted")
        stored = await on_native_loop(client, client._vector_store.get_by_id(output.hy_memory_id))
        assert stored is not None
        assert stored.content == record.verbatim and stored.owner == output.owner
    for record in records:
        assert supervisor.evidence.get(record.evidence_id) == record
    raw = await on_native_loop(
        client,
        client._vector_store.list_by_user(
            user_id="local",
            layers=[MemoryLayer.L1_RAW],
            allow_raw_layer=True,
        ),
    )
    allowed_raw = [node for node in raw if node.node_id not in DENIED_IDS]
    assert allowed_raw
    # One raw copy per turn; together they hold every record.
    for record in records if raw_records is None else raw_records:
        line = f"[{record.role}]: {record.verbatim}"
        assert any(line in node.content for node in allowed_raw), line
    with sqlite3.connect(supervisor.data_dir / "one-memory.sqlite3") as db:
        proofs = db.execute("SELECT node_id, evidence_json FROM hy_node_permissions").fetchall()
    assert {output.hy_memory_id for output in outputs} <= {key for key, _ in proofs}
    assert not set(DENIED_IDS) & {key for key, _ in proofs}
    allowed_refs = [evidence_ref(record) for record in records]
    for _, refs in proofs:
        assert json.loads(refs) and all(ref in allowed_refs for ref in json.loads(refs))
    return {output.hy_memory_id for output in outputs}


def record_receipt(test_node, supervisor, task, oracle, *, phase, hidden):
    test_node.user_properties.append(
        (
            phase,
            json.dumps(
                {
                    "sdk": hy_memory_version(),
                    "transport": "scripted-http-oracle-not-real-model",
                    "request_count": len(oracle.requests),
                    "response_limit": oracle.limit,
                    "stages": oracle.stages,
                    "oracle_errors": oracle.errors,
                    "requests_after_revocation": len(oracle.requests) - oracle.revoked_at
                    if oracle.revoked_at is not None
                    else None,
                    "hidden_native_fixture_checked": hidden,
                    "status": task.status,
                    "error": task.last_error_code,
                    "outputs": [
                        output.model_dump(mode="json")
                        for output in supervisor.tasks.outputs(task.task_id)
                    ],
                    "governance_count": len(supervisor.governance.list(user_id="local")),
                }
            ),
        )
    )


@pytest.mark.parametrize("hidden", [False, True], ids=["fresh-raw", "denied-native-context"])
def test_native_raw_input_runs_extraction_guards_and_governance(tmp_path, hidden, request):
    test_node = request.node

    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, _):
            tasks = supervisor.prepare_conversation(records[0])
            assert len(tasks) == 1
            task = tasks[0]
            runner = supervisor._create_runner()
            _, request, _ = runner._prepare_permission_lease(task)
            client = await native_client(supervisor, task)
            before = await seed_denied(client, request) if hidden else {}
            oracle = ScriptedOracle(records)
            bind_oracle(supervisor, oracle)
            result = await supervisor._create_runner().run(task.task_id)
            assert not oracle.errors, oracle.errors
            if hidden:
                assert await node_snapshots(client, DENIED_IDS) == before
            assert result.status == "completed", (result.last_error_code, oracle.stages)
            assert oracle.stages.count("extract") == 1
            assert "review" not in oracle.stages and "citations" not in oracle.stages
            assert "embedding" in oracle.stages
            assert "reconcile" not in oracle.stages
            await assert_outputs(supervisor, task, records, client)
            record_receipt(test_node, supervisor, result, oracle, phase="raw", hidden=hidden)

    asyncio.run(scenario())


def test_a_tool_turn_hy_judges_as_chat_is_extracted_without_its_tool_messages(tmp_path):
    async def scenario():
        async with raw_client(tmp_path, tool="[Bash] pytest -q\n12 passed") as (
            supervisor,
            records,
            _,
        ):
            tasks = supervisor.prepare_conversation(records[0])
            assert len(tasks) == 1
            oracle = ScriptedOracle(records)
            bind_oracle(supervisor, oracle)
            result = await supervisor._create_runner().run(tasks[0].task_id)
            assert not oracle.errors, oracle.errors
            assert result.status == "completed", (result.last_error_code, oracle.stages)
            assert oracle.stages[0] == "judge"
            prompt = next(
                body["messages"][-1]["content"]
                for _path, body in oracle.requests
                if "messages" in body and "## New Messages\n" in body["messages"][-1]["content"]
            )
            turns = prompt_json(prompt, "## New Messages\n")
            assert [(turn["role"], turn["content"]) for turn in turns] == [
                (record.role, record.verbatim) for record in records[:2]
            ]
            client = await native_client(supervisor, tasks[0])
            await assert_outputs(supervisor, tasks[0], records, client, raw_records=records[:2])

    asyncio.run(scenario())


def test_revoke_during_native_extractor_response_stops_children_and_governance(tmp_path, request):
    test_node = request.node

    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, permission):
            task = supervisor.prepare_conversation(records[0])[0]
            _, request, _ = supervisor._create_runner()._prepare_permission_lease(task)
            client = await native_client(supervisor, task)
            before = await seed_denied(client, request)

            def revoke(stage):
                if stage != "extract":
                    return False
                supervisor.model_content_permissions.revoke(
                    user_id="local", expected_revision=permission.revision, user_confirmed=True
                )
                return True

            oracle = ScriptedOracle(records, after_response=revoke)
            bind_oracle(supervisor, oracle)
            result = await supervisor._create_runner().run(task.task_id)
            assert not oracle.errors, oracle.errors
            assert result.status == "cancelled"
            assert oracle.revoked_at is not None and len(oracle.requests) == oracle.revoked_at
            assert oracle.stages.count("extract") == 1
            assert "review" not in oracle.stages and "citations" not in oracle.stages
            assert not supervisor.governance.list(user_id="local")
            assert not supervisor.tasks.outputs(task.task_id)
            assert await node_snapshots(client, DENIED_IDS) == before
            for record in records:
                assert supervisor.evidence.get(record.evidence_id) == record
            record_receipt(test_node, supervisor, result, oracle, phase="revoked", hidden=True)
            assert result.last_error_code == "model_content_permission_revoked", result

    asyncio.run(scenario())


def test_approved_native_candidates_are_reused_without_disclosing_denied_nodes(tmp_path, request):
    test_node = request.node

    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, permission):
            first = supervisor.prepare_conversation(records[0])[0]
            _, request, _ = supervisor._create_runner()._prepare_permission_lease(first)
            client = await native_client(supervisor, first)
            before = await seed_denied(client, request)
            first_oracle = ScriptedOracle(records)
            bind_oracle(supervisor, first_oracle)
            first_result = await supervisor._create_runner().run(first.task_id)
            assert not first_oracle.errors, first_oracle.errors
            assert first_result.status == "completed", first_result.last_error_code
            ids = await assert_outputs(supervisor, first, records, client)
            record_receipt(
                test_node, supervisor, first_result, first_oracle, phase="initial", hidden=True
            )
            added = put_turn(supervisor.evidence, "user-repeat", "user", USER_TEXT, 3)
            records = (*records, added)
            grant_records(supervisor, records, permission.revision)
            second = supervisor.prepare_conversation(added)[0]
            assert second.task_id != first.task_id
            oracle = ScriptedOracle(records)
            bind_oracle(supervisor, oracle)
            result = await supervisor._create_runner().run(second.task_id)
            assert not oracle.errors, oracle.errors
            assert await node_snapshots(client, DENIED_IDS) == before
            assert result.status == "completed", (result.last_error_code, oracle.stages)
            # SDK 2.0.2 keeps the `limit=1` pre-check removed, so
            # skip reconcile, so the second write now compares against the
            # already-approved candidates instead of blindly ADDing them. The
            # contract is the reuse itself, not the absence of the call: one
            # reconcile pass, the returning ids are the ones already governed,
            # and no third/fourth memory appears.
            assert oracle.stages.count("reconcile") == 1
            assert {item["memory_id"] for item in oracle.reconciles[0]} == ids
            assert await assert_outputs(supervisor, second, records, client) == ids
            assert len(supervisor.governance.list(user_id="local")) == 2
            record_receipt(test_node, supervisor, result, oracle, phase="reuse", hidden=True)

    asyncio.run(scenario())
