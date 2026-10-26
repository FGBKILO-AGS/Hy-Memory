/** The v2 memory-write shapes shared by the MCP tool and the broker route.
 *
 * Core `POST /api/v2/memory-writes` accepts OpenAI-style messages. These
 * checks mirror its `SdkMemoryWriteInput` so a malformed call is refused
 * before a round trip. `source_id` is never part of a host's request: the
 * broker sets it from the authenticated installation, so a model cannot write
 * under another host's name.
 */

export interface MemoryWriteMessage {
  role: "user" | "assistant" | "tool";
  content: string;
}

/** What a model may pass to `MemHub_memory_write`. */
export interface MemoryWriteToolRequest {
  request_id?: string;
  messages: MemoryWriteMessage[];
  memory_at?: string;
  priority?: "live" | "normal";
}

/** What the MCP session sends to the broker: the session id is its own. */
export interface MemoryWriteRequest extends MemoryWriteToolRequest {
  request_id: string;
  session_id: string;
}

const RFC3339 =
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/i;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function boundedString(value: unknown, max: number): value is string {
  return typeof value === "string" && value.length >= 1 && value.length <= max;
}

function hasOnly(value: Record<string, unknown>, keys: readonly string[]) {
  return Object.keys(value).every((key) => keys.includes(key));
}

function validOptionalFields(value: Record<string, unknown>): boolean {
  if (value.request_id !== undefined && !boundedString(value.request_id, 256))
    return false;
  if (
    value.memory_at !== undefined &&
    (typeof value.memory_at !== "string" ||
      !RFC3339.test(value.memory_at) ||
      Number.isNaN(Date.parse(value.memory_at)))
  )
    return false;
  if (
    value.priority !== undefined &&
    value.priority !== "live" &&
    value.priority !== "normal"
  )
    return false;
  const messages = value.messages;
  return (
    Array.isArray(messages) &&
    messages.length >= 1 &&
    messages.length <= 200 &&
    messages.every(
      (message) =>
        isRecord(message) &&
        hasOnly(message, ["role", "content"]) &&
        (message.role === "user" ||
          message.role === "assistant" ||
          message.role === "tool") &&
        boundedString(message.content, 12_000),
    ) &&
    messages.some((message) => message.role === "user")
  );
}

export function isMemoryWriteToolRequest(
  value: unknown,
): value is MemoryWriteToolRequest {
  return (
    isRecord(value) &&
    hasOnly(value, ["request_id", "messages", "memory_at", "priority"]) &&
    validOptionalFields(value)
  );
}

export function isMemoryWriteRequest(
  value: unknown,
): value is MemoryWriteRequest {
  return (
    isRecord(value) &&
    hasOnly(value, [
      "request_id",
      "session_id",
      "messages",
      "memory_at",
      "priority",
    ]) &&
    boundedString(value.request_id, 256) &&
    boundedString(value.session_id, 512) &&
    validOptionalFields(value)
  );
}

export function isMemoryWriteRequestId(value: unknown): value is string {
  return boundedString(value, 256);
}
