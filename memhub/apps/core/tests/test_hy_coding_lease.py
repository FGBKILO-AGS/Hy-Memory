"""HY's own coding/working route under the permission lease, on the pinned SDK.

HY decides the route; scripted HTTP stands in for the model, not for HY.
"""

import asyncio
import json
import sqlite3
import traceback
from collections.abc import Mapping

import httpx
from hy_memory.coding.types import CodingMemory

from one_memory.memory_governance import MemoryKind, MemoryStatus
from tests.test_hy_permission_raw_context import (
    bind_oracle,
    grant_records,
    native_client,
    on_native_loop,
    put_turn,
    raw_client,
)
from tests.test_hy_permission_store import Lease

TOOL_TEXT = "[Bash] pytest -q\n12 passed"
TASK = "Run the synthetic test suite"
SEARCH_KEY = "pytest quiet run"
SOLUTION = "Run `pytest -q` from the repository root."
STATEMENT = "The agent ran pytest -q in the synthetic repository and 12 tests passed."
DENIED_EXPERIENCE = "denied-coding-experience"
DENIED_TASK = "DENIED_CODING_TASK_5d2e run the suite"
DENIED_SOLUTION = "DENIED_CODING_SOLUTION_7f10: skip the tests"

JUDGE = "You are a single-label scene classifier."
EXTRACT = "You extract durable WORKING memories"
RECONCILE = "You decide how to merge new coding memory drafts"
FACT_RECONCILE = "You maintain session-level WORKING FACTS"


def coding_vector():
    # One shared direction: every stored experience and fact is a candidate.
    return [float(index == 5) for index in range(384)]


class FixedEmbeddings:
    async def embed_batch(self, texts):
        return [coding_vector() for _ in texts]

    async def embed(self, text):
        return coding_vector()


class CodingOracle:
    def __init__(
        self,
        records,
        *,
        solution=SOLUTION,
        statement=STATEMENT,
        update_target=None,
        forbidden=(),
        after_response=None,
    ):
        self.records = records
        self.solution, self.statement = solution, statement
        self.update_target = update_target
        self.forbidden = forbidden
        self.after_response = after_response
        self.requests = []
        self.stages = []
        self.reconcile_prompts = []
        self.errors = []
        self.revoked_at = None

    def __call__(self, request):
        try:
            return self.respond(request)
        except Exception:
            self.errors.append(traceback.format_exc())
            raise

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        assert self.revoked_at is None, "native child request escaped after revocation"
        serialized = json.dumps(body, ensure_ascii=False)
        for forbidden in self.forbidden:
            assert forbidden not in serialized, f"unapproved coding row reached HTTP: {forbidden}"
        if request.url.path == "/v1/embeddings":
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            self.stages.append("embedding")
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {"index": i, "object": "embedding", "embedding": coding_vector()}
                        for i in range(len(texts))
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        prompt = body["messages"][-1]["content"]
        if prompt.startswith(JUDGE):
            result = {"is_coding": True, "reason": "synthetic tool run"}
            stage = "judge"
        elif prompt.startswith(EXTRACT):
            for record in self.records:
                assert record.verbatim in prompt, record.role
            result = {
                "experiences": [
                    {
                        "task": TASK,
                        "search_keys": [SEARCH_KEY],
                        "solution": self.solution,
                        "boundary_envs": "",
                        "confidence": 0.9,
                    }
                ],
                "fact": {
                    "subject": "synthetic repository",
                    "statement": self.statement,
                    "files": [],
                    "details": "",
                },
            }
            stage = "extract"
        elif prompt.startswith(RECONCILE):
            self.reconcile_prompts.append(prompt)
            op = (
                {"action": "UPDATE", "draft_idx": 0, "target_experience_id": self.update_target}
                if self.update_target
                else {"action": "ADD", "draft_idx": 0}
            )
            result = {"ops": [op]}
            stage = "reconcile"
        elif prompt.startswith(FACT_RECONCILE):
            result = {"ops": []}
            stage = "fact_reconcile"
        else:
            raise AssertionError(f"unexpected native prompt: {prompt[:80]!r}")
        self.stages.append(stage)
        response = httpx.Response(
            200,
            json={
                "id": "synthetic-coding-response",
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


def proof_ids(supervisor):
    with sqlite3.connect(supervisor.data_dir / "one-memory.sqlite3") as db:
        return {row[0] for row in db.execute("SELECT node_id FROM hy_node_permissions")}


async def run_coding_turn(supervisor, seed, oracle):
    # Renewed consent also re-queues earlier turns; run only the seed's turn.
    (task,) = (
        task
        for task in supervisor.prepare_conversation(seed)
        if seed.evidence_id in {ref["evidence_id"] for ref in supervisor.tasks.inputs(task.task_id)}
    )
    bind_oracle(supervisor, oracle)
    result = await supervisor._create_runner().run(task.task_id)
    assert not oracle.errors, oracle.errors
    return task, result


def by_kind(supervisor):
    memories = supervisor.governance.list(user_id="local")
    grouped = {}
    for memory in memories:
        grouped.setdefault(memory.kind, []).append(memory)
    return grouped


def test_hy_routes_a_tool_turn_to_coding_and_both_layers_are_kept(tmp_path):
    async def scenario():
        async with raw_client(tmp_path, tool=TOOL_TEXT) as (supervisor, records, _):
            oracle = CodingOracle(records)
            task, result = await run_coding_turn(supervisor, records[0], oracle)
            assert result.status == "completed", (result.last_error_code, oracle.stages)
            assert oracle.stages[:2] == ["judge", "extract"]
            assert "reconcile" in oracle.stages
            kinds = by_kind(supervisor)
            assert set(kinds) == {MemoryKind.CODING, MemoryKind.WORKING}
            (coding,), (working,) = kinds[MemoryKind.CODING], kinds[MemoryKind.WORKING]
            assert coding.content == f"{TASK}\n{SOLUTION}"
            assert working.content == STATEMENT
            every_record = {record.evidence_id for record in records}
            for memory in (coding, working):
                assert set(memory.evidence_ids) == every_record
                assert memory.status is MemoryStatus.CONFIRMED
            outputs = supervisor.tasks.outputs(task.task_id)
            assert {output.layer for output in outputs} == {"coding_experience", "working_fact"}
            assert {output.hy_memory_id for output in outputs} <= proof_ids(supervisor)
            # HY's pipeline log copies every coding prompt it is handed a cache for.
            (log_dir,) = tmp_path.rglob("hy-memory/*/logs")
            for path in log_dir.rglob("*"):
                if path.is_file():
                    for text in (records[0].verbatim, TASK):
                        assert text.encode() not in path.read_bytes(), (path.name, text)

    asyncio.run(scenario())


def test_an_unapproved_experience_never_reaches_the_model_or_changes(tmp_path):
    async def scenario():
        async with raw_client(tmp_path, tool=TOOL_TEXT) as (supervisor, records, _):
            task = supervisor.prepare_conversation(records[0])[0]
            _, request, _ = supervisor._create_runner()._prepare_permission_lease(task)
            client = await native_client(supervisor, task)

            async def seed():
                await client._get_coding_writer()
                store = client._coding_store
                native_embeddings = store.embed_service
                store.embed_service = FixedEmbeddings()
                try:
                    await store.insert(
                        CodingMemory(
                            experience_id=DENIED_EXPERIENCE,
                            user_id=request.user_id,
                            agent_id=request.agent_id,
                            session_id=request.session_id,
                            task=DENIED_TASK,
                            solution=DENIED_SOLUTION,
                        )
                    )
                finally:
                    store.embed_service = native_embeddings
                found = await store.search_by_query_embedding(
                    coding_vector(), user_id=request.user_id
                )
                assert [hit["experience_id"] for hit in found] == [DENIED_EXPERIENCE]
                return (await store.get_by_id(DENIED_EXPERIENCE)).to_dict()

            before = await on_native_loop(client, seed())
            oracle = CodingOracle(
                records, forbidden=(DENIED_EXPERIENCE, DENIED_TASK, DENIED_SOLUTION)
            )
            bind_oracle(supervisor, oracle)
            result = await supervisor._create_runner().run(task.task_id)
            assert not oracle.errors, oracle.errors
            assert result.status == "completed", (result.last_error_code, oracle.stages)
            assert oracle.reconcile_prompts
            after = await on_native_loop(client, client._coding_store.get_by_id(DENIED_EXPERIENCE))
            assert after.to_dict() == before
            assert DENIED_EXPERIENCE not in proof_ids(supervisor)
            contents = {memory.content for memory in supervisor.governance.list(user_id="local")}
            assert contents == {f"{TASK}\n{SOLUTION}", STATEMENT}

    asyncio.run(scenario())


def test_a_later_turn_updating_an_experience_updates_the_same_memory(tmp_path):
    async def scenario():
        async with raw_client(tmp_path, tool=TOOL_TEXT) as (supervisor, records, permission):
            _, first = await run_coding_turn(supervisor, records[0], CodingOracle(records))
            assert first.status == "completed", first.last_error_code
            (original,) = by_kind(supervisor)[MemoryKind.CODING]
            question = put_turn(
                supervisor.evidence, "user-again", "user", "Run it again verbosely.", 4
            )
            tool = put_turn(
                supervisor.evidence, "tool-again", "tool", "[Bash] pytest -v\n12 passed", 5
            )
            everything = (*records, question, tool)
            grant_records(supervisor, everything, permission.revision)
            verbose = "Run `pytest -v` from the repository root for per-test output."
            oracle = CodingOracle(
                (question, tool),
                solution=verbose,
                statement="The agent re-ran the suite verbosely; 12 passed.",
                update_target=original.memory_id,
            )
            _, second = await run_coding_turn(supervisor, question, oracle)
            assert second.status == "completed", (second.last_error_code, oracle.stages)
            assert original.memory_id in oracle.reconcile_prompts[0]
            kinds = by_kind(supervisor)
            (updated,) = kinds[MemoryKind.CODING]
            assert updated.memory_id == original.memory_id
            assert updated.content == f"{TASK}\n{verbose}"
            assert updated.revision > original.revision
            assert {question.evidence_id, tool.evidence_id} <= set(updated.evidence_ids)
            assert len(kinds[MemoryKind.WORKING]) == 2

    asyncio.run(scenario())


def test_revoking_during_coding_extraction_stops_the_route(tmp_path):
    async def scenario():
        async with raw_client(tmp_path, tool=TOOL_TEXT) as (supervisor, records, permission):

            def revoke(stage):
                if stage != "extract":
                    return False
                supervisor.model_content_permissions.revoke(
                    user_id="local", expected_revision=permission.revision, user_confirmed=True
                )
                return True

            oracle = CodingOracle(records, after_response=revoke)
            task, result = await run_coding_turn(supervisor, records[0], oracle)
            assert result.status == "cancelled"
            assert result.last_error_code == "model_content_permission_revoked", result
            assert oracle.revoked_at is not None and len(oracle.requests) == oracle.revoked_at
            assert not supervisor.governance.list(user_id="local")
            assert not supervisor.tasks.outputs(task.task_id)

    asyncio.run(scenario())


def test_exposed_coding_config_is_a_redacted_projection_not_the_native_handle(tmp_path):
    """Public SDK 2.0.2 reads `config` off the coding store; it must not hand out secrets.

    The wrapper answers `config` so the SDK's routing works, but the native
    MemoryConfig is a live backend handle carrying api keys and passwords. This
    asserts callers get a plain projection: not the native object, and no
    credential value survives anywhere inside it.
    """

    async def scenario():
        async with raw_client(tmp_path, tool=TOOL_TEXT) as (supervisor, records, _):
            task = supervisor.prepare_conversation(records[0])[0]
            client = await native_client(supervisor, task)
            await on_native_loop(client, client._get_coding_writer())
            return client._coding_store

    native = asyncio.run(scenario())
    from one_memory.hy_permission_coding_store import HyPermissionCodingStore

    native.config.llm.api_key = "DENIED_LLM_KEY_2f31"
    native.config.embedder.api_key = "DENIED_EMBED_KEY_9c04"
    native.config.graph_store.password = "DENIED_GRAPH_PW_77aa"
    store = HyPermissionCodingStore(
        native,
        Lease(),
        agent_id="agent",
        session_id="session",
    )

    exposed = store.config
    assert exposed is not native.config
    assert not isinstance(exposed, type(native.config))
    # Attribute access still works: that is how the public SDK reads it.
    assert isinstance(exposed["llm"], Mapping)
    assert exposed["llm"]["model"] == native.config.llm.model
    assert exposed.llm.model == native.config.llm.model

    def contains_secret(value, depth=0):
        if depth > 5:
            return False
        if isinstance(value, str):
            return "DENIED_" in value
        if isinstance(value, Mapping):
            return any(contains_secret(item, depth + 1) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_secret(item, depth + 1) for item in value)
        return False

    assert not contains_secret(exposed)
    assert exposed["llm"]["api_key"] is None
    assert exposed["embedder"]["api_key"] is None
    assert exposed["graph_store"]["password"] is None
    # No native handle, connection or callable escapes the projection.
    assert not hasattr(exposed, "_native")
    assert "native" not in exposed
