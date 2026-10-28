"""DeepSeek Harness has two profiles; only the headless session form is importable.

The source matrix states it plainly: headless is `<session>/session.jsonl` with a
`type=session` header; web is not a Harness session file and must go through the
`*-export` family instead. That distinction was prose with no test behind it, so
this pins the observable boundary: a headless session imports, while a web-style
export dropped into the same `sessions/` tree imports nothing.
"""

import asyncio
import json
from pathlib import Path

from one_memory.imports.grants import SourceGrantStore
from one_memory.imports.jobs import ImportJobStore, ImportStatus
from one_memory.imports.sources.registry import RootCandidateKind, build_source_catalog
from one_memory.imports.storage import EvidenceStore
from one_memory.imports.supervisor import ImportSupervisor

HEADLESS_RECORDS = [
    {"type": "session", "id": "s", "cwd": "/private/project"},
    {
        "type": "user/message",
        "seq": 1,
        "data": {
            "id": "a",
            "role": "user",
            "source": {"kind": "user"},
            "content": [{"type": "text", "text": "headless person turn"}],
        },
    },
]

# The shape a browser-side export has: a `role`-keyed transcript, no Harness
# `type=session` envelope, and not named `session.jsonl`.
WEB_EXPORT_RECORDS = [
    {"role": "user", "content": "web export person turn", "time": None},
    {"role": "assistant", "content": "web export agent turn", "time": None},
]


class Source:
    def __init__(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        self.registration = build_source_catalog(home=home, environment={})["deepseek-harness"]
        self.root = self.registration.candidate.path
        self.root.mkdir(parents=True, exist_ok=True)
        database = tmp_path / "data" / "core.sqlite3"
        database.parent.mkdir(parents=True, exist_ok=True)
        self.jobs = ImportJobStore(database)
        self.evidence = EvidenceStore(database)
        self.grants = SourceGrantStore(database, home=home)
        self.grants.grant(
            user_id="local",
            source_id="deepseek-harness",
            root=self.root,
            candidate_kind=RootCandidateKind.DEFAULT,
        )

    def write(self, relative: str, records: list[dict]) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        return path

    def run(self):
        async def scenario():
            supervisor = ImportSupervisor(
                adapters={
                    "deepseek-harness": self.registration.create_adapter(
                        self.grants.authorized_root(user_id="local", source_id="deepseek-harness")
                    )
                },
                evidence_store=self.evidence,
                job_store=self.jobs,
            )
            job = supervisor.create_job("deepseek-harness")
            supervisor.enqueue(job.job_id)
            return await supervisor.wait(job.job_id)

        return asyncio.run(scenario())


def test_only_the_headless_session_form_is_importable(tmp_path):
    fixture = Source(tmp_path)
    fixture.write("headless-run/session.jsonl", HEADLESS_RECORDS)
    # A web export dropped into the same tree: different name, different shape.
    fixture.write("web-export/transcript.jsonl", WEB_EXPORT_RECORDS)

    result = fixture.run()
    assert result.status is ImportStatus.COMPLETED
    assert result.stored == 1, (
        "only the headless session.jsonl may import; the web export must be "
        f"skipped, got stored={result.stored}"
    )
    stored = [row.verbatim for row in fixture.evidence.list_extractable(limit=50)]
    assert "headless person turn" in stored
    assert not any("web export" in text for text in stored), (
        "a web-style export must not be imported through the Harness profile"
    )


def test_a_web_export_alone_imports_nothing(tmp_path):
    fixture = Source(tmp_path)
    fixture.write("web-export/transcript.jsonl", WEB_EXPORT_RECORDS)

    result = fixture.run()
    assert result.status is ImportStatus.COMPLETED
    assert result.stored == 0
    assert fixture.evidence.count() == 0


def test_the_adapter_has_two_independent_layers(tmp_path):
    """File-name gating and record-shape gating are separate defences.

    Mutation M28 (relax `file_predicate` to accept any `.jsonl`) did NOT turn the
    tests above red, and that is honest rather than a weak assertion: the parser
    rejects a record with no `data` envelope as UNKNOWN_RECORD, so the web export
    still imports nothing. This test states both layers explicitly so neither is
    mistaken for the other, and so a future change that removes one is visible.
    """
    fixture = Source(tmp_path)
    adapter = fixture.registration.create_adapter(
        fixture.grants.authorized_root(user_id="local", source_id="deepseek-harness")
    )
    headless = fixture.write("headless-run/session.jsonl", HEADLESS_RECORDS)
    web = fixture.write("web-export/transcript.jsonl", WEB_EXPORT_RECORDS)

    # Layer 1: the file predicate accepts only the headless session filename.
    predicate = adapter.file_predicate
    listed_headless = adapter.grant.listed_file(headless)
    listed_web = adapter.grant.listed_file(web)
    assert predicate(listed_headless) is True
    assert predicate(listed_web) is False, (
        "the file predicate must exclude a web-style export by name"
    )

    # Layer 2: even if a web record reached the parser, it is skipped, not stored
    # as a person's words.
    from one_memory.imports.sources.common import SourceRecordContext
    from one_memory.imports.sources.deepseek_harness import parse_deepseek_harness_record

    context = SourceRecordContext(
        file_key="k",
        relative_path="web-export/transcript.jsonl",
        ordinal="k:1",
        header=None,
    )
    parsed = parse_deepseek_harness_record(WEB_EXPORT_RECORDS[0], context)
    assert not hasattr(parsed, "verbatim"), (
        "a record without the Harness data envelope must not become Evidence"
    )


# Harness 0.2.x writes the same session envelope into a *versioned* file name
# (`session.v3.jsonl.zstd` / `session.v4.jsonl.zstd`), and in v4 the tool output
# left the `tool-result` block: `toolCallId`/`isError` moved onto the message and
# the payload became a plain `text` block. Both changes were silent: the file name
# gate dropped whole sessions, and the block gate dropped every tool result.
V4_RECORDS = [
    {
        "type": "session",
        "version": 4,
        "id": "s4",
        "cwd": "/private/project",
        "isSeeded": False,
        "delegationDepth": 0,
    },
    {
        "type": "user/message",
        "seq": 1,
        "data": {
            "id": "u4",
            "role": "user",
            "source": {"kind": "user"},
            "content": [{"type": "text", "text": "v4 person turn"}],
        },
    },
    {
        "type": "tool/result",
        "time": 1_700_000_000,
        "data": {
            "turn": "t1",
            "step": "st1",
            "message": {
                "role": "user",
                "source": {"kind": "tool"},
                "toolCallId": "call_00_x",
                "isError": False,
                "id": "tr4",
                "content": [{"type": "text", "text": "v4 tool output"}],
            },
        },
    },
]

LEGACY_TOOL_RESULT_RECORDS = [
    V4_RECORDS[0],
    {
        "type": "tool/result",
        "time": 1_700_000_000,
        "data": {
            "turn": "t1",
            "step": "st1",
            "message": {
                "role": "user",
                "source": {"kind": "tool"},
                "id": "tr4",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "call_00_x",
                        "content": "legacy tool output",
                        "isError": False,
                    }
                ],
            },
        },
    },
]


def write_zstd(root, relative: str, records: list[dict]) -> Path:
    import zstandard

    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record) + "\n" for record in records).encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(payload))
    return path


def test_a_multi_digit_version_bump_still_imports(tmp_path):
    """The version segment is `v<digits>`, not a single digit.

    This one is guarded by mutation: narrowing `[0-9]+` to `[0-9]` turns it red.
    An earlier version of this test asserted that a rotated `session.jsonl.bak-…`
    copy stays out; it kept passing under every loosening of the pattern (the
    extension gate rejects that name before the file predicate runs, and
    `re.match` anchors the head on its own), so it pinned nothing and was
    dropped rather than kept as a green decoration.
    """
    fixture = Source(tmp_path)
    # The envelope carries its own version, so a `v10` file name must not be
    # paired with a `version: 4` header — that mismatch is a fixture bug, not a
    # gate under test.
    header10 = {**V4_RECORDS[0], "version": 10}
    write_zstd(
        fixture.root,
        "future-run/session.v10.jsonl.zstd",
        [header10, *V4_RECORDS[1:]],
    )

    result = fixture.run()
    assert result.status is ImportStatus.COMPLETED
    assert result.stored == 2, (
        f"a two-digit version bump must still be read, got stored={result.stored}"
    )


def test_the_versioned_session_file_name_imports(tmp_path):
    """A v4 session must import by name, not be dropped as unknown.

    Mutation: this is the `file_predicate` layer alone — the same records read
    from `session.jsonl` already import, so a failure here is the name, not the
    parser.
    """
    fixture = Source(tmp_path)
    write_zstd(fixture.root, "v4-run/session.v4.jsonl.zstd", V4_RECORDS)

    result = fixture.run()
    assert result.status is ImportStatus.COMPLETED
    assert result.stored == 2, (
        f"a `session.v4.jsonl.zstd` must import its turns, got stored={result.stored}"
    )
    stored = [row.verbatim for row in fixture.evidence.list_extractable(limit=50)]
    assert "v4 person turn" in stored


def test_a_v4_tool_result_reaches_evidence(tmp_path):
    """v4 moved the payload out of the `tool-result` block; it must not vanish.

    Parsed directly, so the file-name gate cannot mask or cause this.
    """
    from one_memory.imports.sources.common import SourceRecordContext
    from one_memory.imports.sources.deepseek_harness import (
        parse_deepseek_harness_record,
    )

    context = SourceRecordContext(
        file_key="k",
        relative_path="v4-run/session.v4.jsonl.zstd",
        ordinal="k:2",
        header=V4_RECORDS[0],
    )
    parsed = parse_deepseek_harness_record(V4_RECORDS[2], context)
    assert getattr(parsed, "verbatim", None) == "v4 tool output", (
        "a v4 `tool/result` whose content is a plain text block must still become "
        f"tool evidence, got {parsed!r}"
    )

    # The older shape must keep working, unchanged, beside it.
    legacy = parse_deepseek_harness_record(
        LEGACY_TOOL_RESULT_RECORDS[1],
        SourceRecordContext(
            file_key="k2",
            relative_path="old-run/session.jsonl",
            ordinal="k2:1",
            header=LEGACY_TOOL_RESULT_RECORDS[0],
        ),
    )
    assert "legacy tool output" in legacy.verbatim


def test_one_session_split_across_two_files_stores_each_message_once(tmp_path):
    """Harness upgraded mid-session, so one conversation lives in two files.

    Widening the name gate must not mint a second Evidence row for the turn both
    files carry: identity is `conversation_id + message_id`, not the file.
    """
    fixture = Source(tmp_path)
    shared = {
        "type": "user/message",
        "seq": 1,
        "data": {
            "id": "shared-turn",
            "role": "user",
            "source": {"kind": "user"},
            "content": [{"type": "text", "text": "turn written before the upgrade"}],
        },
    }
    header = V4_RECORDS[0]
    fixture.write("split-run/session.jsonl", [header, shared])
    write_zstd(
        fixture.root,
        "split-run/session.v4.jsonl.zstd",
        [header, shared, V4_RECORDS[1]],
    )

    result = fixture.run()
    assert result.status is ImportStatus.COMPLETED
    assert fixture.evidence.count() == 2, (
        "two files, three records with a header and a turn each: the shared turn "
        "must store once and the v4-only turn once, got "
        f"{fixture.evidence.count()}"
    )
    stored = sorted(row.verbatim for row in fixture.evidence.list_extractable(limit=50))
    assert stored.count("turn written before the upgrade") == 1
    assert stored == sorted(["turn written before the upgrade", "v4 person turn"])
