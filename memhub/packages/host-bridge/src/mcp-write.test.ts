// A3-02: the MCP write tools.
//
// The point of these tools is not that they exist, but that they hand back a
// durable id instead of writing synchronously. A host that thinks the write
// completed on return would resubmit on the next turn, and HY has no
// idempotency key (A0-03), so a resubmit would duplicate the memory.

import { describe, expect, it, vi } from "vitest";

import { processMcpLine } from "./mcp-protocol.js";
import { McpSession, type MemoryWriteRequest } from "./mcp-session.js";

/** Build a session whose only wired tools are the write ones.
 *
 * The read tools are given stubs that reject, so a test that accidentally
 * dispatches to one fails loudly instead of silently returning nothing. Only
 * the two tools under test are injected.
 */
function sessionWith(
  write: (path: string, request: MemoryWriteRequest) => Promise<unknown>,
  status: (path: string, id: string) => Promise<unknown>,
): McpSession {
  const unwired = async () => {
    throw new Error("read tool is not wired in this test");
  };
  return new McpSession(
    "/tmp/discovery.json",
    unwired as never,
    undefined,
    unwired as never,
    unwired as never,
    unwired as never,
    write as never,
    status as never,
  );
}

async function initialize(session: McpSession) {
  await processMcpLine(
    session,
    JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2025-03-26", capabilities: {} },
    }),
  );
}

describe("om_memory_write MCP tool", () => {
  it("is listed alongside the read tools", async () => {
    const session = sessionWith(
      async () => ({ request_id: "r1", state: "accepted" }),
      async () => null,
    );
    await initialize(session);
    const listed = await processMcpLine(
      session,
      JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list" }),
    );
    const tools = JSON.parse(listed ?? "{}").result.tools as {
      name: string;
    }[];
    expect(tools.map((tool) => tool.name)).toContain("MemHub_memory_write");
    expect(tools.map((tool) => tool.name)).toContain(
      "MemHub_memory_write_status",
    );
  });

  const call = (id: number, args: unknown) =>
    JSON.stringify({
      jsonrpc: "2.0",
      id,
      method: "tools/call",
      params: { name: "MemHub_memory_write", arguments: args },
    });
  const messages = [{ role: "user", content: "remember tabs over spaces" }];

  it("returns the durable request id instead of writing synchronously", async () => {
    const write = vi.fn(async (_path: string, request: MemoryWriteRequest) => ({
      request_id: request.request_id,
      state: "accepted",
      created_at: "2026",
      updated_at: "2026",
      detail: "",
    }));
    const session = sessionWith(write, async () => null);
    await initialize(session);

    const response = await processMcpLine(
      session,
      call(3, { messages, request_id: "req-42" }),
    );
    const payload = JSON.parse(response ?? "{}").result.structuredContent as {
      request_id: string;
      state: string;
    };
    expect(payload.request_id).toBe("req-42");
    expect(payload.state).toBe("accepted");
  });

  it("generates the request id and owns the session id", async () => {
    const write = vi.fn(async () => ({ request_id: "r", state: "accepted" }));
    const session = sessionWith(write, async () => null);
    await initialize(session);

    await processMcpLine(session, call(3, { messages }));
    await processMcpLine(session, call(4, { messages, priority: "live" }));
    const [first, second] = write.mock.calls.map(
      (args) => (args as unknown as [string, MemoryWriteRequest])[1],
    );
    expect(first?.request_id).toMatch(/^[0-9a-f-]{36}$/);
    expect(second?.request_id).not.toBe(first?.request_id);
    expect(first?.session_id).toMatch(/^mcp-[0-9a-f-]{36}$/);
    expect(second?.session_id).toBe(first?.session_id);
    expect(first?.messages).toEqual(messages);
    expect(second?.priority).toBe("live");
  });

  it("does not let the model name the session or the source", async () => {
    const write = vi.fn(async () => ({ request_id: "never" }));
    const session = sessionWith(write, async () => null);
    await initialize(session);

    for (const extra of [{ session_id: "s" }, { source_id: "claude-code" }]) {
      const response = await processMcpLine(
        session,
        call(5, { messages, ...extra }),
      );
      expect(JSON.parse(response ?? "{}").error.code).toBe(-32602);
    }
    expect(write).not.toHaveBeenCalled();
  });

  it("rejects a write without a user turn before any round trip", async () => {
    const write = vi.fn(async () => ({ request_id: "never" }));
    const session = sessionWith(write, async () => null);
    await initialize(session);

    for (const args of [
      { messages: [] },
      { messages: [{ role: "assistant", content: "hi" }] },
      { messages: [{ role: "system", content: "hi" }] },
      { messages, memory_at: "yesterday" },
    ]) {
      const response = await processMcpLine(session, call(6, args));
      expect(JSON.parse(response ?? "{}").error.code).toBe(-32602);
    }
    expect(write).not.toHaveBeenCalled();
  });

  it("passes a never-accepted status through as null rather than an error", async () => {
    const status = vi.fn(async () => null);
    const session = sessionWith(async () => ({ request_id: "r" }), status);
    await initialize(session);

    const response = await processMcpLine(
      session,
      JSON.stringify({
        jsonrpc: "2.0",
        id: 6,
        method: "tools/call",
        params: {
          name: "MemHub_memory_write_status",
          arguments: { request_id: "unknown-id" },
        },
      }),
    );
    expect(status).toHaveBeenCalledWith("/tmp/discovery.json", "unknown-id");
    const result = JSON.parse(response ?? "{}").result;
    expect(result.isError).toBeFalsy();
    expect(result.structuredContent).toBeNull();
  });
});
