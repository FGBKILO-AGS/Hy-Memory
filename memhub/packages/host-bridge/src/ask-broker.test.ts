import { mkdtemp } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import os from "node:os";
import path from "node:path";
import axios from "axios";
import { afterEach, describe, expect, it } from "vitest";

import { RecallBroker } from "./recall-broker.js";

const closers: Array<() => Promise<void>> = [];

afterEach(async () => {
  await Promise.allSettled(closers.splice(0).map((close) => close()));
});

describe("RecallBroker /ask", () => {
  it("proxies ask to Core and returns empty refuse verbatim", async () => {
    let authorization: string | undefined;
    let readContext: string | string[] | undefined;
    let coreBody: unknown;
    const core = createServer(async (request, response) => {
      authorization = request.headers.authorization;
      readContext = request.headers["x-one-memory-read-context"];
      if (request.url !== "/api/v2/memories/ask") {
        response.statusCode = 404;
        response.end("not found");
        return;
      }
      const chunks: Buffer[] = [];
      request.on("data", (chunk) => chunks.push(chunk));
      request.on("end", () => {
        coreBody = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        response.setHeader("Content-Type", "application/json");
        response.end(
          JSON.stringify({
            schema_version: 2,
            answer: "现有记忆不足以回答这个问题。",
            memories: [],
            mode: "empty",
          }),
        );
      });
    });
    const corePort = await listen(core, 0);
    closers.push(() => close(core));
    const root = await mkdtemp(path.join(os.tmpdir(), "om-ask-broker-"));
    const discoveryPath = path.join(root, "cursor-broker.json");
    const broker = new RecallBroker({
      coreBaseUrl: `http://127.0.0.1:${corePort}`,
      coreApiToken: "master-secret",
      discoveryPath,
      instanceId: "ask-instance",
      preferredPort: 0,
    });
    await broker.start();
    closers.push(() => broker.stop());

    const response = await axios.post(
      `http://127.0.0.1:${broker.port}/ask`,
      {
        question: "我下周的航班是什么？",
        scope: "all-apps",
        workspace_path: "/tmp/project",
      },
      {
        headers: {
          "Content-Type": "application/json",
          "X-One-Memory-Client": "cursor-mcp",
        },
      },
    );

    expect(authorization).toBe("Bearer master-secret");
    expect(readContext).toBe("external");
    expect(coreBody).toEqual({
      question: "我下周的航班是什么？",
      scope: "all-apps",
      workspace_path: "/tmp/project",
    });
    expect(response.data).toEqual({
      schema_version: 2,
      answer: "现有记忆不足以回答这个问题。",
      memories: [],
      mode: "empty",
    });
  });

  it("rejects a legacy-shaped answer instead of passing it to the host", async () => {
    const core = createServer(async (request, response) => {
      for await (const _chunk of request) void _chunk;
      response.setHeader("Content-Type", "application/json");
      response.end(
        JSON.stringify({ answer: "legacy", memories: [], mode: "empty" }),
      );
    });
    const corePort = await listen(core, 0);
    closers.push(() => close(core));
    const root = await mkdtemp(path.join(os.tmpdir(), "om-ask-legacy-"));
    const broker = new RecallBroker({
      coreBaseUrl: `http://127.0.0.1:${corePort}`,
      coreApiToken: "master-secret",
      discoveryPath: path.join(root, "broker.json"),
      instanceId: "ask-legacy",
      preferredPort: 0,
    });
    await broker.start();
    closers.push(() => broker.stop());

    const response = await axios.post(
      `http://127.0.0.1:${broker.port}/ask`,
      { question: "我的偏好？" },
      {
        headers: { "X-One-Memory-Client": "cursor-mcp" },
        validateStatus: () => true,
      },
    );

    expect(response.status).toBe(502);
  });

  it("asks Core v2 directly, with no authority round trip or Governance fields", async () => {
    const calls: string[] = [];
    const core = createServer(async (request, response) => {
      calls.push(request.url ?? "");
      response.setHeader("Content-Type", "application/json");
      if (request.url === "/api/v2/memories/ask") {
        for await (const _chunk of request) void _chunk;
        response.end(
          JSON.stringify({
            schema_version: 2,
            answer: "你偏好本地优先。",
            memories: [
              {
                memory_id: "memory-1",
                content: "用户偏好本地优先。",
                layer: "l2_fact",
                status: "active",
                source: "cursor",
                gmt_created: null,
              },
            ],
            mode: "extractive",
          }),
        );
        return;
      }
      response.statusCode = 404;
      response.end("not found");
    });
    const corePort = await listen(core, 0);
    closers.push(() => close(core));
    const root = await mkdtemp(path.join(os.tmpdir(), "om-ask-sdk-"));
    const broker = new RecallBroker({
      coreBaseUrl: `http://127.0.0.1:${corePort}`,
      coreApiToken: "master-secret",
      discoveryPath: path.join(root, "broker.json"),
      instanceId: "ask-sdk",
      preferredPort: 0,
    });
    await broker.start();
    closers.push(() => broker.stop());

    const response = await axios.post(
      `http://127.0.0.1:${broker.port}/ask`,
      { question: "我的偏好？" },
      { headers: { "X-One-Memory-Client": "cursor-mcp" } },
    );

    expect(calls).toEqual(["/api/v2/memories/ask"]);
    expect(response.data.schema_version).toBe(2);
    expect(JSON.stringify(response.data)).not.toMatch(
      /revision|evidence_ids|scopes/,
    );
  });
});

function listen(
  server: ReturnType<typeof createServer>,
  port: number,
): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      resolve((server.address() as AddressInfo).port);
    });
  });
}

function close(server: ReturnType<typeof createServer>): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}
