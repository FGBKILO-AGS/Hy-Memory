import re
from pathlib import PurePosixPath

from pydantic import JsonValue

from one_memory.imports.access import AuthorizedSourceRoot, safe_project_label
from one_memory.imports.adapters import (
    ScanIssueCode,
    SourceCapabilities,
    SourceDescriptor,
    StorageKind,
)
from one_memory.imports.models import WrittenBy
from one_memory.imports.sources.common import (
    JsonlSourceAdapter,
    ParsedMessage,
    ParsedSkip,
    SourceRecordContext,
)
from one_memory.imports.sources.parsing import (
    as_record,
    first_string,
    parse_timestamp,
    text_content,
    tool_blocks_evidence,
    tool_evidence_text,
    tool_text_evidence,
)

#: Harness renames the session file when it bumps the record format, so the same
#: envelope arrives as `session.jsonl`, `session.jsonl.zstd`, `session.v3.jsonl.zstd`,
#: `session.v4.jsonl.zstd`, … Gating on the two original names dropped every
#: session written by a newer Harness, silently and in bulk.
SESSION_FILE_NAME = re.compile(r"^session(\.v[0-9]+)?\.jsonl(\.zstd)?$")

DEEPSEEK_HARNESS_DESCRIPTOR = SourceDescriptor(
    source_id="deepseek-harness",
    display_name="DeepSeek Harness",
    storage_kinds=frozenset({StorageKind.JSONL, StorageKind.ZSTD}),
    capabilities=SourceCapabilities(
        can_append=True,
        can_observe_edits=False,
        can_observe_deletes=False,
        can_reconcile_complete_set=False,
    ),
)


def create_deepseek_harness_adapter(
    grant: AuthorizedSourceRoot,
    *,
    descriptor: SourceDescriptor = DEEPSEEK_HARNESS_DESCRIPTOR,
    source_version: str | None = None,
) -> JsonlSourceAdapter:
    return JsonlSourceAdapter(
        descriptor=descriptor,
        grant=grant,
        parser=parse_deepseek_harness_record,
        source_version=source_version,
        file_predicate=lambda source_file: bool(
            SESSION_FILE_NAME.match(source_file.relative_path.name)
        ),
        suffixes=frozenset({".jsonl", ".zstd"}),
        header_selector=_is_session_header,
    )


def _is_session_header(record: dict[str, JsonValue]) -> bool:
    """Harness writes the session's `cwd` once, in the first line only.

    Verified against a session this machine produced: the header is
    `{"type":"session","version":0,"id":...,"cwd":"/private/tmp/dsh-verify"}`
    and no later record repeats it. Read per-record, `cwd` is therefore always
    absent and every turn imports with no workspace.
    """
    return record.get("type") == "session"


def parse_deepseek_harness_record(
    record: dict[str, JsonValue],
    context: SourceRecordContext,
) -> ParsedMessage | ParsedSkip:
    relative = PurePosixPath(context.relative_path)
    conversation_id = (
        first_string((context.header or {}).get("id"))
        or relative.parent.name
        or relative.name.split(".", maxsplit=1)[0]
    )
    event_type = record.get("type")
    data = as_record(record.get("data"))
    if event_type == "session":
        return ParsedSkip(reason_code=ScanIssueCode.UNKNOWN_RECORD)
    if data is None:
        return ParsedSkip(
            reason_code=ScanIssueCode.UNKNOWN_RECORD,
            conversation_id=conversation_id,
        )
    workspace = first_string(record.get("cwd"), (context.header or {}).get("cwd"))
    if event_type in {"tool/call", "tool/result"}:
        return _tool_message(record, data, conversation_id, workspace)
    message: dict[str, JsonValue] | None
    if event_type == "user/message":
        source = as_record(data.get("source"))
        if source is None or source.get("kind") != "user":
            return ParsedSkip(
                reason_code=ScanIssueCode.INTERNAL_REASONING_EXCLUDED,
                conversation_id=conversation_id,
            )
        message = data
    elif event_type == "assistant/message":
        message = as_record(data.get("message"))
        if message is None:
            return ParsedSkip(
                reason_code=ScanIssueCode.UNKNOWN_RECORD,
                conversation_id=conversation_id,
            )
    else:
        return ParsedSkip(
            reason_code=ScanIssueCode.INTERNAL_REASONING_EXCLUDED,
            conversation_id=conversation_id,
        )
    role = first_string(message.get("role"))
    if role not in {"user", "assistant"}:
        return ParsedSkip(
            reason_code=ScanIssueCode.TOOL_CONTENT_EXCLUDED,
            conversation_id=conversation_id,
        )
    text = text_content(
        message.get("content"),
        allowed_block_types=frozenset({"text"}),
    )
    if not text:
        return ParsedSkip(
            reason_code=ScanIssueCode.NO_TEXT,
            conversation_id=conversation_id,
        )
    return ParsedMessage(
        conversation_id=conversation_id,
        message_id=first_string(message.get("id"), message.get("messageId")),
        role=role,
        written_by=WrittenBy.PERSON if role == "user" else WrittenBy.AGENT,
        created_at=parse_timestamp(record.get("time") or message.get("timestamp")),
        verbatim=text,
        workspace=workspace,
        project=safe_project_label(workspace),
        record_type=str(event_type),
    )


def _tool_message(
    record: dict[str, JsonValue],
    data: dict[str, JsonValue],
    conversation_id: str,
    workspace: str | None,
) -> ParsedMessage | ParsedSkip:
    """`tool/call` carries name and arguments; `tool/result` a user-role message.

    The same call also appears as a `tool-call` block in `assistant/message`;
    only the event is read, so it is stored once.
    """
    message = as_record(data.get("message"))
    if record.get("type") == "tool/result" and message is not None:
        text = tool_blocks_evidence(message.get("content"), block_types=frozenset({"tool-result"}))
        if text is None:
            # v4 dropped the wrapper: the same result arrives as plain text
            # blocks, with `toolCallId`/`isError` moved onto the message.
            text = tool_text_evidence(message.get("content"))
        message_id = first_string(message.get("id"), data.get("callId"))
    else:
        text = tool_evidence_text(data)
        message_id = first_string(data.get("callId"), data.get("id"))
    if text is None:
        return ParsedSkip(
            reason_code=ScanIssueCode.TOOL_CONTENT_EXCLUDED,
            conversation_id=conversation_id,
        )
    return ParsedMessage(
        conversation_id=conversation_id,
        message_id=message_id,
        role="tool",
        written_by=WrittenBy.TOOL,
        created_at=parse_timestamp(record.get("time")),
        verbatim=text,
        workspace=workspace,
        project=safe_project_label(workspace),
        record_type=str(record.get("type")),
    )
