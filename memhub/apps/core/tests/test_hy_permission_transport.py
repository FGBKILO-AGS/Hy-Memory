"""Native HY entry points, synthetic inputs and MockTransport only."""

import asyncio
import copy
import inspect
import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from hy_memory.agent.llm_provider import LLMProvider, LLMResponse, LLMRetryError
from hy_memory.config import MemoryConfig
from hy_memory.core.embed_service import EmbedService
from one_memory.model_content_permissions import (
    ModelContentPermissionReceipt,
    ModelContentPermissionSnapshot,
    _recipients,
)

from one_memory.hy_permission_transport import HyPermissionTransport, HyPermissionTransportError
from one_memory.hy_response_guard import HyResponseGuard
from one_memory.hy_runtime_config import HY_MEMORY_DISTRIBUTION, HY_MEMORY_VERSION
from one_memory.model_profiles import ModelProfileSnapshot, ProviderKind, attach_disable_thinking


@dataclass
class Lease:
    receipt: ModelContentPermissionReceipt
    profile: ModelProfileSnapshot
    checks: int = 0
    revoked: bool = False
    context_changed: bool = False

    def recheck(self):
        self.checks += 1
        if self.revoked or self.context_changed:
            raise RuntimeError("private permission error must not escape")
        return self.receipt


def lease(*, local=False, thinking=False):
    from one_memory.embedding_assets import BUILTIN_EMBEDDING_DIMENSIONS, BUILTIN_EMBEDDING_MODEL

    profile = ModelProfileSnapshot(
        profile_id="synthetic-profile",
        revision=1,
        provider_kind=ProviderKind.OPENAI_COMPATIBLE,
        base_url="https://models.example.test/approved/v1",
        model="synthetic-generation",
        embedding_provider_kind=ProviderKind.LOCAL_ONNX
        if local
        else ProviderKind.OPENAI_COMPATIBLE,
        embedding_base_url=None if local else "https://models.example.test/embedding/v1",
        embedding_model=BUILTIN_EMBEDDING_MODEL if local else "synthetic-embedding",
        embedding_dimensions=BUILTIN_EMBEDDING_DIMENSIONS if local else 2,
        timeout_seconds=5,
        max_concurrency=1,
        context_window=None,
        has_api_key=True,
        disable_thinking=thinking,
    )
    generation, embedding = _recipients(profile)
    now = datetime.now(UTC)
    snapshot = ModelContentPermissionSnapshot(
        user_id="synthetic-user",
        profile=profile,
        generation=generation,
        embedding=embedding,
        sources=[
            dict(
                source_id="one-memory-manual",
                authority="manual",
                granted_at=now,
                revision=1,
                root_identity=None,
                root_path="",
                requested_path=None,
                candidate_kind="default",
            )
        ],
        evidence=[
            dict(
                evidence_id="synthetic-evidence",
                source_id="one-memory-manual",
                revision=1,
                content_hash="a" * 64,
                identity_hash="b" * 64,
                origin_hash="c" * 64,
                imported_at=now,
            )
        ],
    )
    return Lease(
        ModelContentPermissionReceipt(
            user_id=snapshot.user_id,
            revision=1,
            revocation_revision=0,
            granted_at=now,
            revoked_at=None,
            snapshot=snapshot,
        ),
        profile,
    )


def native(permission):
    config = MemoryConfig()
    config.llm.provider = "openai"
    config.llm.base_url = permission.receipt.snapshot.generation.endpoint
    config.llm.model = permission.profile.model
    config.llm.api_key = "synthetic-secret-never-log"
    config.llm.max_retries = 2
    config.llm.retry_delay = 0
    config.llm.extra_headers = None
    config.llm.extra_body = attach_disable_thinking(permission.profile, {}) or None
    config.embedder.provider = "openai"
    config.embedder.base_url = (
        permission.receipt.snapshot.embedding.endpoint or "http://127.0.0.1/local-onnx"
    )
    config.embedder.model = permission.profile.embedding_model
    config.embedder.api_key = "synthetic-secret-never-log"
    config.embedder.embedding_dims = permission.profile.embedding_dimensions
    config.embedder.max_retries = 2
    config.embedder.retry_delay = 0
    config.embedder.extra_headers = None
    config.embedder.extra_body = None
    provider, embedder = LLMProvider(config), EmbedService(config)
    embedder._embed_config.queue_batch_window_ms = 1
    return provider, embedder


def response(request):
    if request.url.path.endswith("/embeddings"):
        inputs = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "synthetic-embedding",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.25, 0.75]}
                    for i, _ in enumerate(inputs)
                ],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
            headers={"x-private-header": "private-response-metadata"},
        )
    return httpx.Response(
        200,
        json={
            "id": "synthetic-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "synthetic-generation",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "native result",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "record",
                                    "arguments": '{"value":"unaltered"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
        },
    )


class Wire(httpx.MockTransport):
    def __init__(self, handler=None):
        self.requests = []
        self.closed = 0
        self.callback = handler
        super().__init__(self.send)

    async def send(self, request):
        self.requests.append(request)
        return self.callback(request, len(self.requests)) if self.callback else response(request)

    async def aclose(self):
        self.closed += 1
        await super().aclose()


def test_native_entry_points_preserve_payload_response_and_restore_instances():
    async def run():
        permission = lease(thinking=True)
        provider, embedder = native(permission)
        original = dict(provider.__dict__)
        complete_signature = inspect.signature(provider.complete)
        wire = Wire()
        messages = [
            {"role": "system", "content": "exact system"},
            {"role": "user", "content": "exact user"},
        ]
        before = copy.deepcopy(messages)
        tools = [{"type": "function", "function": {"name": "record", "parameters": {}}}]
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            assert guard.bind_provider(provider) is provider
            assert inspect.signature(provider.complete) == complete_signature
            guard.bind_embed_service(embedder)
            saved_complete = provider.complete
            result = await provider.complete("exact native prompt\n  spaces", stop=["STOP"])
            assert isinstance(result, LLMResponse)
            assert (
                result.content,
                result.tokens_used,
                result.prompt_tokens,
                result.completion_tokens,
                result.finish_reason,
            ) == ("native result", 11, 7, 4, "tool_calls")
            assert result.tool_calls[0]["function"]["arguments"] == '{"value":"unaltered"}'
            await provider.complete_messages(messages, tools=tools, tool_choice="auto")
            # Native dispatch itself also converges on the guarded leaf.
            await LLMProvider._call_llm(provider, "direct native dispatch", 12, 0)
            assert await embedder.embed("synthetic text", use_cache=False) == [0.25, 0.75]
            vector, meta = await embedder.embed_with_meta("another", use_cache=False)
            assert vector == [0.25, 0.75]
            assert set(meta) <= {"http_status", "model", "usage", "attempts", "cache_hit"}
            assert "private-response-metadata" not in repr(meta)
            assert permission.checks >= len(wire.requests)
        assert wire.closed == 1
        for name in ("complete", "complete_messages", "_call_llm", "_chat_completions"):
            assert provider.__dict__.get(name) is original.get(name)
        with pytest.raises(HyPermissionTransportError, match="ended"):
            await saved_complete("must not send")
        assert len(wire.requests) == 5
        bodies = [json.loads(request.content) for request in wire.requests]
        assert bodies[0]["messages"] == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "exact native prompt\n  spaces"},
        ]
        assert bodies[0]["stop"] == ["STOP"]
        assert bodies[0]["reasoning_effort"] == "none"
        assert bodies[1]["messages"] == before == messages
        assert bodies[1]["tools"] == tools and bodies[1]["tool_choice"] == "auto"
        assert str(wire.requests[3].url).endswith("/embedding/v1/embeddings")
        assert bodies[3]["dimensions"] == 2

    asyncio.run(run())


def test_permission_transport_pins_the_public_sdk_identity():
    """The transport fails closed unless the audited SDK identity is installed."""
    assert HY_MEMORY_DISTRIBUTION == "hy-memory"
    assert HY_MEMORY_VERSION == "2.0.2"


@pytest.mark.parametrize("entry", ["complete", "complete_messages", "embed"])
@pytest.mark.parametrize("revoked", [False, True])
def test_retry_rechecks_each_send_and_latches_across_fresh_child_providers(entry, revoked):
    async def run():
        permission = lease()
        provider, embedder = native(permission)

        def handler(request, count):
            if count == 1:
                permission.revoked = revoked
                if entry == "embed":
                    raise httpx.ReadTimeout("private timeout details", request=request)
                return httpx.Response(500, json={"error": {"message": "private failure"}})
            return response(request)

        wire = Wire(handler)
        guard = HyPermissionTransport(permission, http_transport=wire)
        try:
            async with guard:
                guard.bind_provider(provider)
                guard.bind_embed_service(embedder)
                call = (
                    provider.complete("prompt")
                    if entry == "complete"
                    else provider.complete_messages([{"role": "user", "content": "prompt"}])
                    if entry == "complete_messages"
                    else embedder.embed("text")
                )
                if not revoked:
                    await call
                    assert len(wire.requests) == 2
                else:
                    with pytest.raises(HyPermissionTransportError):
                        await call
                    checks = permission.checks
                    permission.revoked = False
                    child, _ = native(permission)
                    with pytest.raises(HyPermissionTransportError):
                        guard.bind_provider(child)
                    with pytest.raises(HyPermissionTransportError):
                        await provider._call_llm("child", 10, 0)
                    assert permission.checks == checks
                    assert len(wire.requests) == 1
        except HyPermissionTransportError:
            assert revoked  # Exit must also refuse a swallowed denial.
        assert wire.closed == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "entry,expected", [("complete", 2), ("complete_messages", 2), ("embed", 3)]
)
def test_native_outer_attempt_budgets_no_openai_hidden_retries(entry, expected, caplog):
    """0.2.0 keeps retry ownership outside OpenAI's client: our leaf sends each attempt.

    The expected counts are intentionally unchanged from the old SDK pin: 0.2.0
    must not hide retries below this transport, or D84's per-send permission and
    Cookie/header checks would be bypassed.
    """

    async def run():
        permission = lease()
        provider, embedder = native(permission)

        def fail(request, _):
            raise httpx.ConnectError("private upstream body and secret", request=request)

        wire = Wire(fail)
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_provider(provider)
            guard.bind_embed_service(embedder)
            with pytest.raises((LLMRetryError, HyPermissionTransportError)) as error:
                if entry == "embed":
                    await embedder.embed("private synthetic input")
                elif entry == "complete":
                    await provider.complete("private synthetic input")
                else:
                    await provider.complete_messages([{"role": "user", "content": "input"}])
            assert "private" not in str(error.value)
            assert len(wire.requests) == expected
            if entry != "embed":
                assert len(error.value.attempts) == expected
                assert provider._errors == expected
        assert "private" not in caplog.text

    asyncio.run(run())


@pytest.mark.parametrize("entry", ["complete", "complete_messages"])
@pytest.mark.parametrize(
    "failure,kind",
    [
        (httpx.ReadTimeout, "timeout"),
        (httpx.ConnectTimeout, "timeout"),
        (httpx.WriteTimeout, "timeout"),
        (httpx.PoolTimeout, "timeout"),
        (httpx.ConnectError, "connection"),
    ],
)
def test_generation_failure_category_survives_sanitized_native_attempts(
    entry, failure, kind, caplog
):
    """0.2.0 still exposes sanitized attempt records with our five failure kinds.

    Timeout and connection categories must stay distinguishable after the SDK
    upgrade; otherwise real provider failures collapse into an unknown retry path.
    """

    async def run():
        permission = lease()
        provider, _ = native(permission)

        def fail(request, _count):
            raise failure("private upstream body secret must not escape", request=request)

        wire = Wire(fail)
        async with HyPermissionTransport(permission, http_transport=wire) as transport:
            transport.bind_provider(provider)
            guard = HyResponseGuard(provider)
            with pytest.raises(LLMRetryError) as raised:
                if entry == "complete":
                    await guard.complete("private input")
                else:
                    await guard.complete_messages([{"role": "user", "content": "private input"}])
            assert len(wire.requests) == 2
            assert len(raised.value.attempts) == 2
            assert all(item["failure_kind"] == kind for item in raised.value.attempts)
            assert all("status_code" not in item for item in raised.value.attempts)
            expected = (
                "hy_memory_model_timeout" if kind == "timeout" else "hy_memory_model_request_failed"
            )
            assert guard.failures == [expected]
            assert "private" not in json.dumps(raised.value.attempts)
            assert "private" not in caplog.text
            assert "synthetic-secret-never-log" not in caplog.text
        assert wire.closed == 1

    asyncio.run(run())


@pytest.mark.parametrize("entry", ["complete", "complete_messages"])
def test_malformed_generation_response_is_not_classified_as_timeout(entry):
    async def run():
        permission = lease()
        provider, _ = native(permission)
        wire = Wire(lambda _request, _count: httpx.Response(200, json={"choices": []}))
        async with HyPermissionTransport(permission, http_transport=wire) as transport:
            transport.bind_provider(provider)
            guard = HyResponseGuard(provider)
            with pytest.raises(LLMRetryError) as raised:
                if entry == "complete":
                    await guard.complete("synthetic input")
                else:
                    await guard.complete_messages([{"role": "user", "content": "synthetic input"}])
            assert len(wire.requests) == 2
            assert all(item["failure_kind"] == "invalid_response" for item in raised.value.attempts)
            assert guard.failures == ["hy_memory_model_request_failed"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    ["host", "path", "query", "model", "backend", "dimensions", "headers", "body", "key", "kwargs"],
)
def test_config_drift_or_overrides_deny_before_send(change):
    async def run():
        permission = lease()
        provider, embedder = native(permission)
        wire = Wire()
        with pytest.raises(HyPermissionTransportError):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_provider(provider)
                guard.bind_embed_service(embedder)
                if change in {"host", "path", "query"}:
                    provider._llm_config.base_url += {
                        "host": ".evil.test",
                        "path": "/other",
                        "query": "?token=private",
                    }[change]
                elif change == "model":
                    provider._llm_config.model = "not-approved"
                elif change == "backend":
                    # Public SDK 2.0.2 keeps LLMBackend narrowed to OPENAI, so the
                    # drifted-backend case uses an illegal sentinel value
                    # instead. It must differ from the approved openai backend,
                    # otherwise no drift is constructed at all.
                    provider._llm_config.backend = "eval-platform"
                elif change == "dimensions":
                    embedder._embed_config.embedding_dims = 3
                elif change == "headers":
                    provider._llm_config.extra_headers = {"Authorization": "not-approved"}
                elif change == "body":
                    provider._llm_config.extra_body = {"model": "not-approved"}
                elif change == "key":
                    provider._llm_config.api_key = "changed"
                if change == "dimensions":
                    await embedder.embed("input")
                else:
                    await provider.complete(
                        "input", **({"model": "override"} if change == "kwargs" else {})
                    )
        assert not wire.requests
        assert wire.closed == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "change", ["method", "url", "model", "messages", "dimensions", "auth", "host", "stream"]
)
def test_physical_send_checks_final_request_after_httpx_hooks(change):
    async def run():
        permission = lease()
        provider, embedder = native(permission)
        wire = Wire()

        async def tamper(request):
            if change == "method":
                request.method = "PUT"
            elif change == "url":
                request.url = httpx.URL("https://unexpected.example.test/v1/chat/completions")
            elif change in {"model", "messages", "dimensions"}:
                body = json.loads(request.content)
                body[change] = 3 if change == "dimensions" else "override"
                request._content = json.dumps(body).encode()
                request.stream = httpx.ByteStream(request._content)
            elif change == "stream":
                body = json.loads(request.content)
                body["model"] = "stream-only-override"
                request.stream = httpx.ByteStream(json.dumps(body).encode())
            else:
                request.headers["authorization" if change == "auth" else "host"] = "override"

        with pytest.raises(HyPermissionTransportError):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_provider(provider)
                guard.bind_embed_service(embedder)
                guard._http_client.event_hooks["request"].append(tamper)
                if change == "dimensions":
                    await embedder.embed("input")
                else:
                    await provider.complete("input")
        assert not wire.requests

    asyncio.run(run())


def test_redirect_is_terminal_and_cannot_reach_another_recipient():
    async def run():
        permission = lease()
        provider, _ = native(permission)
        wire = Wire(
            lambda request, _: httpx.Response(
                307,
                headers={
                    "location": "https://unexpected.example.test/steal",
                },
            )
        )
        with pytest.raises(HyPermissionTransportError, match="redirect"):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_provider(provider)
                await provider.complete("input")
        assert len(wire.requests) == 1 and wire.closed == 1

    asyncio.run(run())


def test_lease_revoked_during_successful_response_cannot_publish():
    async def run():
        permission = lease()
        provider, _ = native(permission)

        def revoke(request, _):
            permission.revoked = True
            return response(request)

        wire = Wire(revoke)
        with pytest.raises(HyPermissionTransportError):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_provider(provider)
                await provider.complete("input")
        assert len(wire.requests) == 1

    asyncio.run(run())


def test_queue_and_cache_are_lease_owned_and_outstanding_work_is_cancelled():
    """0.2.0's EmbedService uses an LRU cache, so the lease supplies OrderedDict.

    The old dict still accepted plain assignment, but 0.2.0's cache-hit path
    calls move_to_end; a plain dict here fails the full embedding path and
    masks the real permission/Cookie behavior.
    """

    async def run():
        permission = lease()
        _, embedder = native(permission)
        assert isinstance(embedder._cache, dict)
        old_cache = embedder._cache
        old_cache[embedder._get_cache_key("input")] = [99.0, 99.0]
        wire = Wire()
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_embed_service(embedder)
            assert await embedder.embed_queued("input") == [0.25, 0.75]
            assert len(wire.requests) == 1
            embedder._embed_config.queue_batch_window_ms = 60000
            pending = asyncio.create_task(embedder.embed_queued("pending"))
            await asyncio.sleep(0)
            assert embedder._queue_pending
        assert pending.cancelled()
        assert not embedder._queue_pending
        assert not embedder._queue_flush_tasks
        assert embedder._cache is old_cache
        assert len(wire.requests) == 1 and wire.closed == 1

    asyncio.run(run())


def test_pending_foreign_queue_and_double_install_are_rejected():
    async def run():
        permission = lease()
        provider, embedder = native(permission)
        future = asyncio.get_running_loop().create_future()
        embedder._queue_pending[("https://other.example/v1", "secret")] = [
            ("other lease input", future)
        ]
        wire = Wire()
        with pytest.raises(HyPermissionTransportError, match="queue"):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_embed_service(embedder)
        assert embedder._queue_pending == {
            ("https://other.example/v1", "secret"): [("other lease input", future)]
        }
        async with HyPermissionTransport(permission, http_transport=Wire()) as first:
            first.bind_provider(provider)
            with pytest.raises(HyPermissionTransportError, match="bound"):
                async with HyPermissionTransport(permission, http_transport=Wire()) as second:
                    second.bind_provider(provider)
            await provider.complete("first still works")
        future.cancel()

    asyncio.run(run())


def test_local_onnx_stays_local_and_remote_fallback_is_denied():
    from one_memory.embedding_assets import BUILTIN_EMBEDDING_DIMENSIONS
    from one_memory.hy_memory_bridge import LocalHyEmbedService

    async def run():
        permission = lease(local=True)
        local = LocalHyEmbedService(
            SimpleNamespace(
                encode=lambda texts: [[0.5] * BUILTIN_EMBEDDING_DIMENSIONS for _ in texts],
            )
        )
        wire = Wire()
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_local_embed_service(local)
            assert len(await local.embed("input")) == BUILTIN_EMBEDDING_DIMENSIONS
            saved = local.embed
        with pytest.raises(HyPermissionTransportError, match="ended"):
            await saved("late")
        _, remote = native(permission)
        with pytest.raises(HyPermissionTransportError):
            async with HyPermissionTransport(permission, http_transport=Wire()) as guard:
                guard.bind_embed_service(remote)
        assert not wire.requests

    asyncio.run(run())


def test_sdk_0_2_0_has_no_legacy_provider_specific_bypass_methods():
    """0.2.0 still exposes no provider-specific bypasses on LLMProvider.

    If a new SDK method could send outside complete/complete_messages/_call_llm,
    the transport closure would no longer cover every outbound path.
    """
    permission = lease()
    provider, _ = native(permission)

    assert not hasattr(provider, "_call_openai")
    assert not hasattr(provider, "_call_openai_messages")
    assert not hasattr(provider, "_call_eval_platform")
    assert not hasattr(provider, "_call_eval_platform_messages")


@pytest.mark.parametrize("logger_name", ["openai._base_client", "httpcore.http11"])
def test_wire_debug_logging_does_not_block_provider_calls(caplog, logger_name):
    async def run():
        permission = lease()
        provider, _ = native(permission)
        wire = Wire()
        with caplog.at_level(logging.DEBUG, logger=logger_name):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_provider(provider)
                await provider.complete("private synthetic prompt")
            assert logging.getLogger(logger_name).isEnabledFor(logging.DEBUG)
        assert wire.requests

    asyncio.run(run())


def test_revocation_in_last_request_hook_still_blocks_physical_send():
    async def run():
        permission = lease()
        provider, _ = native(permission)
        wire = Wire()

        async def revoke(_request):
            permission.revoked = True

        with pytest.raises(HyPermissionTransportError):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_provider(provider)
                guard._http_client.event_hooks["request"].append(revoke)
                await provider.complete("input")
        assert not wire.requests and wire.closed == 1

    asyncio.run(run())


def test_unenveloped_http_and_unsupported_leaf_arguments_fail_closed():
    async def run():
        permission = lease()
        provider, _ = native(permission)
        wire = Wire()
        with pytest.raises(HyPermissionTransportError, match="unsupported_request"):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                await guard._http_client.post(
                    permission.receipt.snapshot.generation.endpoint + "/chat/completions",
                    json={"model": permission.profile.model},
                )
        assert not wire.requests
        with pytest.raises(HyPermissionTransportError, match="unsupported_operation"):
            async with HyPermissionTransport(permission, http_transport=Wire()) as guard:
                guard.bind_provider(provider)
                await provider.complete_messages([], 10, 0, stop=["not in native signature"])

    asyncio.run(run())


def test_independent_instances_and_rebinding_do_not_change_global_methods():
    async def run():
        first_lease, second_lease = lease(), lease()
        first, _ = native(first_lease)
        second, _ = native(second_lease)
        class_methods = dict(LLMProvider.__dict__)
        wires = [Wire(), Wire(), Wire()]
        async with HyPermissionTransport(second_lease, http_transport=wires[1]) as second_guard:
            second_guard.bind_provider(second)
            with pytest.raises(HyPermissionTransportError):
                async with HyPermissionTransport(
                    first_lease, http_transport=wires[0]
                ) as first_guard:
                    first_guard.bind_provider(first)
                    saved_first = first.complete
                    await first.complete("first")
                    first_lease.revoked = True
                    await first.complete("denied")
            await second.complete("second still allowed")
        first_lease.revoked = False
        async with HyPermissionTransport(first_lease, http_transport=wires[2]) as fresh:
            fresh.bind_provider(first)
            with pytest.raises(HyPermissionTransportError, match="ended"):
                await saved_first("old captured method")
            await first.complete("new binding")
        assert [len(wire.requests) for wire in wires] == [1, 1, 1]
        assert all(wire.closed == 1 for wire in wires)
        assert dict(LLMProvider.__dict__) == class_methods

    asyncio.run(run())


def test_cache_hit_and_queued_flush_cannot_outlive_revocation():
    async def run():
        permission = lease()
        _, service = native(permission)
        wire = Wire()
        with pytest.raises(HyPermissionTransportError):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_embed_service(service)
                await service.embed("cached input")
                permission.revoked = True
                await service.embed("cached input")
        assert len(wire.requests) == 1

    asyncio.run(run())


def test_exit_cancels_inflight_send_before_restoring_native_methods():
    async def run():
        permission = lease()
        provider, _ = native(permission)
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def slow(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        transport = httpx.MockTransport(slow)
        async with HyPermissionTransport(permission, http_transport=transport) as guard:
            guard.bind_provider(provider)
            child = asyncio.create_task(provider.complete("in flight"))
            await entered.wait()
            http_client = guard._http_client
            sdk_client = guard._clients[0]
        assert child.cancelled() and cancelled.is_set()
        assert http_client.is_closed and sdk_client.is_closed()
        assert provider.complete.__func__ is LLMProvider.complete

    asyncio.run(run())


@pytest.mark.parametrize("field", ["data", "usage"])
def test_malformed_embedding_response_does_not_publish_raw_metadata(field, caplog):
    async def run():
        permission = lease()
        _, service = native(permission)

        def invalid(request, _):
            data = json.loads(response(request).content)
            if field == "data":
                data["data"][0]["embedding"] = ["private-raw-provider-value", 0.5]
            else:
                data["usage"]["prompt_tokens"] = "private-raw-provider-value"
            return httpx.Response(200, json=data)

        wire = Wire(invalid)
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_embed_service(service)
            with pytest.raises(HyPermissionTransportError) as error:
                await service.embed_with_meta("input")
            assert "private-raw" not in str(error.value)
            assert "private-raw" not in repr(service._last_embed_meta)
        assert "private-raw" not in caplog.text

    asyncio.run(run())


def test_local_encoder_failure_is_sanitized_without_cloud_fallback(caplog):
    from one_memory.hy_memory_bridge import LocalHyEmbedService

    async def run():
        permission = lease(local=True)

        def broken(_texts):
            raise RuntimeError("private-raw-encoder-error")

        service = LocalHyEmbedService(SimpleNamespace(encode=broken))
        wire = Wire()
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_local_embed_service(service)
            with pytest.raises(HyPermissionTransportError) as error:
                await service.embed("input")
            assert "private-raw" not in str(error.value)
        assert not wire.requests and "private-raw" not in caplog.text

    asyncio.run(run())


def test_local_encoder_thread_is_drained_before_restoration():
    from one_memory.hy_memory_bridge import LocalHyEmbedService

    async def run():
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def encode(_texts):
            entered.set()
            assert release.wait(5)
            finished.set()
            return [[0.5, 0.5]]

        service = LocalHyEmbedService(SimpleNamespace(encode=encode))
        guard = await HyPermissionTransport(lease(local=True), http_transport=Wire()).__aenter__()
        guard.bind_local_embed_service(service)
        child = asyncio.create_task(service.embed("synthetic input"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            exiting = asyncio.create_task(guard.__aexit__(None, None, None))
            while not child.cancelling():
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not child.done()
            assert not exiting.done()
            assert service.embed.__func__ is not LocalHyEmbedService.embed
        finally:
            release.set()
        await exiting
        assert finished.is_set() and child.cancelled()
        assert service.embed.__func__ is LocalHyEmbedService.embed

    asyncio.run(run())


def test_seal_checks_full_lease_then_exit_allows_own_governance_projection():
    async def run():
        permission = lease()
        provider, embedder = native(permission)
        wire = Wire()
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_provider(provider)
            guard.bind_embed_service(embedder)
            await provider.complete("last SDK call")
            checks = permission.checks
            await guard.seal()
            assert guard.requests_sealed and permission.checks == checks + 1
            checks = permission.checks
            # Represents main's own expected version update AFTER full checking.
            # Main separately checks frozen permissions before its commit.
            permission.context_changed = True
            await guard.seal()  # Idempotent; no full-context check after publication.
            for call in (
                provider.complete("late child"),
                provider._call_llm("late direct dispatch", 10, 0),
                embedder.embed("late embed"),
                guard._http_client.post("https://models.example.test/approved/v1/embeddings"),
            ):
                with pytest.raises(HyPermissionTransportError, match="sealed"):
                    await call
            assert permission.checks == checks
        assert permission.checks == checks and wire.closed == 1
        assert len(wire.requests) == 1

    asyncio.run(run())


def test_seal_does_not_skip_revocation_or_a_swallowed_denial():
    async def run():
        permission = lease()
        provider, _ = native(permission)
        wire = Wire()
        with pytest.raises(HyPermissionTransportError, match="lease_denied"):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                guard.bind_provider(provider)
                await provider.complete("positive")
                permission.revoked = True
                with pytest.raises(HyPermissionTransportError, match="lease_denied"):
                    await guard.seal()
                assert not guard.requests_sealed
                permission.revoked = False
                await guard.seal()
        assert wire.closed == 1

    asyncio.run(run())


def test_seal_refuses_inflight_request_then_succeeds_only_after_join():
    async def run():
        permission = lease()
        provider, _ = native(permission)
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow(request):
            entered.set()
            await release.wait()
            return response(request)

        async with HyPermissionTransport(
            permission, http_transport=httpx.MockTransport(slow)
        ) as guard:
            guard.bind_provider(provider)
            child = asyncio.create_task(provider.complete("in flight"))
            await entered.wait()
            with pytest.raises(HyPermissionTransportError, match="active_requests"):
                await guard.seal()
            assert not guard.requests_sealed
            release.set()
            await child
            await guard.seal()
            assert guard.requests_sealed

    asyncio.run(run())


def test_seal_refuses_pending_native_queue_before_timer_has_started():
    async def run():
        permission = lease()
        _, service = native(permission)
        wire = Wire()
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_embed_service(service)
            child = asyncio.create_task(service.embed_queued("pending"))
            await asyncio.sleep(0)
            assert service._queue_pending
            with pytest.raises(HyPermissionTransportError, match="active_requests"):
                await guard.seal()
            assert not guard.requests_sealed
            await child
            # SDK 0.2.0 leaves the bucket key with a None task after the timer
            # has taken ownership of the batch. Waiting only on live tasks is
            # the same synchronization point the old SDK expressed by removing
            # the key.
            await asyncio.gather(
                *(task for task in service._queue_flush_tasks.values() if task is not None)
            )
            await guard.seal()
        assert len(wire.requests) == 1 and wire.closed == 1

    asyncio.run(run())


def test_exit_joins_queue_timer_even_before_its_bound_coroutine_starts():
    async def run():
        permission = lease()
        _, service = native(permission)
        wire = Wire()
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_embed_service(service)
            child = asyncio.create_task(service.embed_queued("pending"))
            await asyncio.sleep(0)
            timer = next(iter(service._queue_flush_tasks.values()))
            assert timer not in guard._tasks
        assert child.cancelled()
        assert timer.done()
        # A timer that faults only because exit outran its first admission check
        # must be retrieved; otherwise asyncio logs an unhandled task exception.
        assert not timer._log_traceback
        assert not wire.requests and wire.closed == 1

    asyncio.run(run())


@pytest.mark.parametrize("bind", [False, True])
def test_close_failure_is_sanitized_and_cannot_skip_instance_restoration(bind):
    class BrokenClose(Wire):
        async def aclose(self):
            await super().aclose()
            raise RuntimeError("private-close-error")

    async def run():
        permission = lease()
        provider, _ = native(permission)
        wire = BrokenClose()
        with pytest.raises(HyPermissionTransportError, match="cleanup_failed"):
            async with HyPermissionTransport(permission, http_transport=wire) as guard:
                if bind:
                    guard.bind_provider(provider)
                    await provider.complete("positive")
        assert provider.complete.__func__ is LLMProvider.complete
        assert guard._http_client.is_closed
        assert wire.closed == 1

    asyncio.run(run())


def test_a_gateway_cookie_does_not_deny_the_next_call_in_the_task():
    """Gateway cookies must not become model request headers.

    0.2.0's client stack still shares the gateway response with the next call
    in the same task. The transport has to keep the CookieJar policy denying
    every domain so the second physical request remains header-identical.
    """

    def handler(request, count):
        reply = response(request)
        reply.headers["set-cookie"] = "x_host_key_access=abc; path=/; HttpOnly"
        return reply

    async def run():
        permission = lease()
        provider, _embedder = native(permission)
        wire = Wire(handler)
        async with HyPermissionTransport(permission, http_transport=wire) as guard:
            guard.bind_provider(provider)
            await provider.complete("judge")
            await provider.complete_messages([{"role": "user", "content": "extract"}])
        assert len(wire.requests) == 2
        assert all("cookie" not in request.headers for request in wire.requests)

    asyncio.run(run())
