"""Readiness never waits for picking up work an earlier run left unfinished.

A 1.0.4 home with pending imports and write jobs used to resume all of it
before the lifespan yielded, so /health did not answer within the Runtime's
60s budget and the Runtime killed a healthy Core, which then began the same
recovery again. Recovery now runs after readiness.
"""

import asyncio
import threading
import time

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from one_memory import hy_sdk_migration
from one_memory.api import create_app
from one_memory.config import Settings
from one_memory.hy_sdk_migration import MetadataRepairResult
from one_memory.secrets import InMemorySecretStore

pytestmark = pytest.mark.fresh_sdk


def application(tmp_path):
    return create_app(
        Settings(
            data_dir=tmp_path / "data",
            source_home=tmp_path,
            api_token=SecretStr("ready-test"),
            imports_enabled=False,
            automatic_memory_processing=False,
            live_sync_enabled=False,
        ),
        secret_store=InMemorySecretStore(),
    )


def client_for(app):
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer ready-test"},
    )


async def health(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    return response.json()


async def wait_for_recovery(client, *, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        body = await health(client)
        if body["recovery"] in {"done", "failed"}:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError("recovery did not finish")


def test_core_is_ready_while_recovery_is_still_running(tmp_path, monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def slow_repair(**_):
        entered.set()
        release.wait(10)
        return MetadataRepairResult(None, 0, 0, 0)

    monkeypatch.setattr(hy_sdk_migration, "repair_sdk_metadata_from_bundle", slow_repair)

    async def scenario():
        app = application(tmp_path)
        try:
            started = time.monotonic()
            async with app.router.lifespan_context(app), client_for(app) as client:
                # Startup finished although the first recovery step cannot.
                assert time.monotonic() - started < 5
                assert await asyncio.to_thread(entered.wait, 5)
                body = await health(client)
                assert body["status"] == "ok"
                assert body["recovery"] == "running"
                # Nothing after the blocked step has been started yet: the
                # write drain in particular would hold the single writer.
                assert app.state.memory_write_drain_task is None
                # Reads answer during recovery.
                assert (await client.get("/api/v1/sources")).status_code == 200
                release.set()
                body = await wait_for_recovery(client)
                assert body["recovery"] == "done"
                assert body["recovery_failed_steps"] == []
                assert app.state.memory_write_drain_task is not None
        finally:
            release.set()

    asyncio.run(scenario())


def test_a_failed_recovery_step_does_not_stop_readiness_or_later_steps(tmp_path, monkeypatch):
    def broken_repair(**_):
        raise OSError("bundle unreadable")

    monkeypatch.setattr(hy_sdk_migration, "repair_sdk_metadata_from_bundle", broken_repair)

    async def scenario():
        app = application(tmp_path)
        async with app.router.lifespan_context(app), client_for(app) as client:
            body = await wait_for_recovery(client)
            assert body["status"] == "ok"
            assert body["recovery"] == "failed"
            assert body["recovery_failed_steps"] == ["sdk_metadata_repair"]
            # The steps after the failed one still ran.
            assert app.state.memory_write_drain_task is not None

    asyncio.run(scenario())


def test_shutdown_during_recovery_stops_it_without_starting_more(tmp_path, monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def slow_repair(**_):
        entered.set()
        release.wait(10)
        return MetadataRepairResult(None, 0, 0, 0)

    monkeypatch.setattr(hy_sdk_migration, "repair_sdk_metadata_from_bundle", slow_repair)

    async def scenario():
        app = application(tmp_path)
        try:
            async with app.router.lifespan_context(app):
                assert await asyncio.to_thread(entered.wait, 5)
                recovery = app.state.recovery_task
            # Shutdown cancelled recovery before it could start the drain.
            assert recovery.done()
            assert app.state.memory_write_drain_task is None
        finally:
            release.set()

    asyncio.run(scenario())
