import { describe, expect, it, vi } from "vitest";

import type { CoreReader } from "./core-client.js";
import type { CoreOperations } from "./core-operations.js";
import {
  readModelContentPermission,
  setModelContentPermission,
} from "./model-content-permission-flow.js";

describe("model content permission flow", () => {
  it("reads the single final-state boolean", async () => {
    const reader = vi.fn().mockResolvedValue({
      state: "ok",
      data: { allowed: true },
    });

    await expect(
      readModelContentPermission(reader as CoreReader),
    ).resolves.toMatchObject({ state: "ok", allowed: true });
    expect(reader).toHaveBeenCalledWith("/api/v1/model-content-permission");
  });

  it.each([
    ["allow", true],
    ["deny", false],
  ] as const)(
    "sets %s directly without revision or receipt fields",
    async (_, allowed) => {
      const post = vi.fn().mockResolvedValue({
        state: "committed",
        data: { allowed },
      });

      await expect(
        setModelContentPermission({ post } as CoreOperations, allowed),
      ).resolves.toMatchObject({ state: "ok", allowed });
      expect(post).toHaveBeenCalledWith("/api/v1/model-content-permission", {
        allowed,
      });
    },
  );

  it("does not accept a malformed or mismatched success response", async () => {
    const malformed = vi.fn().mockResolvedValue({
      state: "committed",
      data: { allowed: true, revision: 1 },
    });
    await expect(
      setModelContentPermission({ post: malformed } as CoreOperations, true),
    ).resolves.toMatchObject({
      state: "unknown",
      code: "model_content_permission_response_invalid",
    });

    const mismatched = vi.fn().mockResolvedValue({
      state: "committed",
      data: { allowed: false },
    });
    await expect(
      setModelContentPermission({ post: mismatched } as CoreOperations, true),
    ).resolves.toMatchObject({
      state: "unknown",
      code: "model_content_permission_state_mismatch",
    });
  });
});
