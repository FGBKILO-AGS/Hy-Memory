import { beforeEach, describe, expect, it, vi } from "vitest";

import { run } from "./cli.js";

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  put: vi.fn(),
  patch: vi.fn(),
  delete: vi.fn(),
}));
vi.mock("axios", () => ({
  default: { create: vi.fn(() => transport), get: vi.fn() },
}));

const runtime = vi.hoisted(() => ({
  inspectRuntimeStatus: vi.fn(),
  discoverRuntimeAttachment: vi.fn(),
}));
vi.mock("@one-memory/runtime-control", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@one-memory/runtime-control")>()),
  ...runtime,
}));

beforeEach(() => {
  vi.clearAllMocks();
  runtime.inspectRuntimeStatus.mockResolvedValue({
    state: "running",
    code: "runtime_verified",
    instanceId: "instance-1",
    host: "127.0.0.1",
    port: 8765,
    version: "1.2.3",
  });
  runtime.discoverRuntimeAttachment.mockResolvedValue({
    instanceId: "instance-1",
    apiToken: "synthetic-runtime-token",
    host: "127.0.0.1",
    port: 8765,
    servicePid: 4242,
    health: {
      status: "ok",
      service: "one-memory-core",
      version: "1.2.3",
      instance_id: "instance-1",
    },
  });
});

describe("model content permission CLI", () => {
  it("documents only final-state permission and manual extraction commands", async () => {
    const permission = await capturedRun(["help", "permission"]);
    expect(permission.code).toBe(0);
    expect(permission.stdout).toContain("memhub permission allow");
    expect(permission.stdout).toContain("memhub permission deny");
    expect(permission.stdout).not.toMatch(
      /\b(preview|confirm|process|revoke)\b/,
    );
    expect(permission.stdout).not.toContain("revision");
    expect(permission.stdout).not.toContain("receipt");

    const processing = await capturedRun(["help", "processing"]);
    expect(processing.code).toBe(0);
    expect(processing.stdout).toContain("memhub processing extract");
    expect(processing.stdout).not.toMatch(/\b(plan|apply|receipt)\b/);
    expect(processing.stdout).not.toContain("backlog");
  });

  it("reads the final state", async () => {
    transport.get.mockResolvedValue({ status: 200, data: { allowed: true } });

    const result = await capturedRun(["permission", "status", "--json"]);

    expect(result.code).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      state: "ok",
      allowed: true,
    });
    expect(transport.get).toHaveBeenCalledWith(
      "/api/v1/model-content-permission",
      undefined,
    );
  });

  it.each([
    ["allow", true],
    ["deny", false],
  ] as const)("sets %s directly with only allowed", async (verb, allowed) => {
    transport.post.mockResolvedValue({ status: 200, data: { allowed } });

    const result = await capturedRun(["permission", verb, "--json"]);

    expect(result.code).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({ state: "ok", allowed });
    expect(transport.post).toHaveBeenCalledWith(
      "/api/v1/model-content-permission",
      { allowed },
      undefined,
    );
  });

  it.each(["preview", "confirm", "process", "revoke"])(
    "rejects removed %s commands before runtime access",
    async (verb) => {
      const result = await capturedRun(["permission", verb, "--json"]);

      expect(result.code).toBe(2);
      expect(runtime.inspectRuntimeStatus).not.toHaveBeenCalled();
      expect(transport.get).not.toHaveBeenCalled();
      expect(transport.post).not.toHaveBeenCalled();
    },
  );

  it.each(["plan", "apply", "receipt"])(
    "rejects removed processing %s commands before runtime access",
    async (verb) => {
      const result = await capturedRun(["processing", verb, "--json"]);

      expect(result.code).toBe(2);
      expect(runtime.inspectRuntimeStatus).not.toHaveBeenCalled();
      expect(transport.get).not.toHaveBeenCalled();
      expect(transport.post).not.toHaveBeenCalled();
    },
  );
});

async function capturedRun(arguments_: string[]) {
  let stdout = "";
  let stderr = "";
  const stdoutSpy = vi
    .spyOn(process.stdout, "write")
    .mockImplementation((chunk) => {
      stdout += String(chunk);
      return true;
    });
  const stderrSpy = vi
    .spyOn(process.stderr, "write")
    .mockImplementation((chunk) => {
      stderr += String(chunk);
      return true;
    });
  try {
    return {
      code: await run(arguments_, {
        ONE_MEMORY_DATA_DIR: "/synthetic/data",
      }),
      stdout,
      stderr,
    };
  } finally {
    stdoutSpy.mockRestore();
    stderrSpy.mockRestore();
  }
}
