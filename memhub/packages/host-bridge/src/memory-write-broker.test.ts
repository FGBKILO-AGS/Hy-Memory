// The MCP write door through the broker (decision B): authenticated by the
// host credential, not grant-governed, forwarded to Core v2 as a trusted local
// caller, and always filed under the authenticated host's own source id.

import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer, type IncomingHttpHeaders } from "node:http";
import type { AddressInfo } from "node:net";
import os from "node:os";
import path from "node:path";
import axios from "axios";
import { afterEach, describe, expect, it } from "vitest";

import {
  askThroughBroker,
  type BrokerClientOptions,
  memoryWriteStatusThroughBroker,
  memoryWriteThroughBroker,
  readHostCredentialFile,
} from "./broker-client.js";
import { HostGrantStore } from "./host-grants.js";
import { RecallBroker } from "./recall-broker.js";

const roots: string[] = [];
const closers: Array<() => Promise<void>> = [];
const coreToken = "isolated-core-only-token";

afterEach(async () => {
  for (const close of closers.splice(0).reverse()) await close();
  await Promise.all(
    roots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

interface CoreCall {
  method: string | undefined;
  url: string | undefined;
  headers: IncomingHttpHeaders;
  body: unknown;
}

const receipt = {
  request_id: "req-1",
  state: "accepted",
  created_at: "2026-10-26T00:00:00Z",
  updated_at: "2026-10-26T00:00:00Z",
  detail: "",
};

async function fixture(options: { writeStatus?: number } = {}) {
  const root = await mkdtemp(path.join(os.tmpdir(), "om-write-broker-"));
  roots.push(root);
  const store = await HostGrantStore.initialize(path.join(root, "grants"));
  const issued = await store.createInstallation({ hostId: "cursor" });
  // Only a read tool is granted: writes must not depend on the grant.
  await store.approveGrants({
    installationId: issued.receipt.installationId,
    expectedRevision: issued.receipt.revision,
    allowedTools: ["om_ask"],
    allowedScopes: ["all-apps"],
  });
  const credentialFile = path.join(root, "host-credential");
  await writeFile(credentialFile, `${issued.secret}\n`, { mode: 0o600 });
  const calls: CoreCall[] = [];
  const core = createServer((request, response) => {
    void (async () => {
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(Buffer.from(chunk));
      const raw = Buffer.concat(chunks).toString("utf8");
      calls.push({
        method: request.method,
        url: request.url,
        headers: request.headers,
        body: raw ? JSON.parse(raw) : undefined,
      });
      response.setHeader("Content-Type", "application/json");
      if (request.url === "/api/v2/memory-writes") {
        response.statusCode = options.writeStatus ?? 202;
        response.end(
          JSON.stringify(
            response.statusCode === 202 ? receipt : { detail: "conflict" },
          ),
        );
        return;
      }
      if (request.url?.startsWith("/api/v2/memory-writes/")) {
        const known = request.url.endsWith("/req-1");
        response.statusCode = known ? 200 : 404;
        response.end(
          JSON.stringify(
            known
              ? { ...receipt, state: "completed", memory_ids: ["m1"] }
              : { detail: "unknown request id" },
          ),
        );
        return;
      }
      if (request.url === "/api/v2/memories/ask") {
        response.end(
          JSON.stringify({
            schema_version: 2,
            answer: "fixture",
            memories: [],
            mode: "empty",
          }),
        );
        return;
      }
      response.statusCode = 404;
      response.end("{}");
    })();
  });
  await new Promise<void>((resolve) => core.listen(0, "127.0.0.1", resolve));
  closers.push(async () => {
    core.closeAllConnections();
    await new Promise<void>((resolve) => core.close(() => resolve()));
  });
  const discoveryPath = path.join(root, "broker.json");
  const broker = new RecallBroker({
    coreBaseUrl: `http://127.0.0.1:${(core.address() as AddressInfo).port}`,
    coreApiToken: coreToken,
    discoveryPath,
    instanceId: "isolated-write-broker",
    preferredPort: 0,
    hostGrantStore: store,
  });
  await broker.start();
  closers.push(() => broker.stop());
  const client: BrokerClientOptions = {
    credentialProvider: () => readHostCredentialFile(credentialFile),
  };
  return {
    calls,
    discoveryPath,
    client,
    baseUrl: `http://127.0.0.1:${broker.port}`,
  };
}

const write = {
  request_id: "req-1",
  session_id: "mcp-session",
  messages: [{ role: "user" as const, content: "remember tabs over spaces" }],
};

describe("broker memory writes", () => {
  it("forwards to Core v2 as the authenticated host, without the read envelope", async () => {
    const f = await fixture();
    const result = await memoryWriteThroughBroker(
      f.discoveryPath,
      write,
      f.client,
    );
    expect(result.request_id).toBe("req-1");
    const call = f.calls.find((item) => item.url === "/api/v2/memory-writes");
    expect(call?.body).toEqual({ ...write, source_id: "cursor" });
    expect(call?.headers.authorization).toBe(`Bearer ${coreToken}`);
    expect(call?.headers["x-one-memory-read-context"]).toBeUndefined();
    expect(call?.headers["x-one-memory-allowed-tools"]).toBeUndefined();
  });

  it("refuses a body that names its own source", async () => {
    const f = await fixture();
    await expect(
      memoryWriteThroughBroker(
        f.discoveryPath,
        { ...write, source_id: "claude-code" } as never,
        f.client,
      ),
    ).rejects.toMatchObject({ code: "memory_write_invalid" });
    expect(f.calls.some((item) => item.url === "/api/v2/memory-writes")).toBe(
      false,
    );
  });

  it("refuses a write with no user turn before reaching Core", async () => {
    const f = await fixture();
    await expect(
      memoryWriteThroughBroker(
        f.discoveryPath,
        { ...write, messages: [{ role: "assistant", content: "hi" }] },
        f.client,
      ),
    ).rejects.toBeDefined();
    expect(f.calls.some((item) => item.url === "/api/v2/memory-writes")).toBe(
      false,
    );
  });

  it("requires a host credential", async () => {
    const f = await fixture();
    const response = await axios.post(`${f.baseUrl}/memory-writes`, write, {
      headers: { "Content-Type": "application/json" },
      proxy: false,
      validateStatus: () => true,
    });
    expect(response.status).toBe(401);
    expect(f.calls.some((item) => item.url === "/api/v2/memory-writes")).toBe(
      false,
    );
  });

  it("reports a reused request id with other content as a conflict", async () => {
    const f = await fixture({ writeStatus: 409 });
    await expect(
      memoryWriteThroughBroker(f.discoveryPath, write, f.client),
    ).rejects.toMatchObject({ code: "memory_write_conflict" });
  });

  it("reads a status, and returns null for an id Core never accepted", async () => {
    const f = await fixture();
    const known = await memoryWriteStatusThroughBroker(
      f.discoveryPath,
      "req-1",
      f.client,
    );
    expect(known?.state).toBe("completed");
    const unknown = await memoryWriteStatusThroughBroker(
      f.discoveryPath,
      "never-seen",
      f.client,
    );
    expect(unknown).toBeNull();
    const call = f.calls.find(
      (item) => item.url === "/api/v2/memory-writes/req-1",
    );
    expect(call?.method).toBe("GET");
    expect(call?.headers["x-one-memory-read-context"]).toBeUndefined();
  });
});

describe("broker data plane", () => {
  it("asks Core v2 directly, without first asking which data plane it reads", async () => {
    const f = await fixture();
    await expect(
      askThroughBroker(f.discoveryPath, { question: "one" }, f.client),
    ).resolves.toMatchObject({ schema_version: 2, mode: "empty" });
    await askThroughBroker(f.discoveryPath, { question: "two" }, f.client);
    expect(f.calls.map((item) => item.url)).toEqual([
      "/api/v2/memories/ask",
      "/api/v2/memories/ask",
    ]);
  });
});
