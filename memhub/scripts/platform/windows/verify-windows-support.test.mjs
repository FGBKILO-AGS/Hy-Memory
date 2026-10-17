import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

const script = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "verify-windows-support.mjs",
);

describe("Windows support matrix", () => {
  it("reports explicit blocked status until a Windows 11 x64 target is available", () => {
    const result = spawnSync(process.execPath, [script], { encoding: "utf8" });
    assert.equal(result.error, undefined);
    const matrix = JSON.parse(result.stdout);
    assert.deepEqual(matrix.release_gate, {
      os: "Windows 11",
      arch: "x64",
      status: "blocked",
      blocked_reason: "windows_target_machine_required",
    });
    assert.deepEqual(matrix.explicit_matrix, {
      windows_11_x64: "blocked",
      windows_10_x64: "blocked",
      windows_arm64: "blocked",
    });
    assert.equal(matrix.capabilities.app_cli_shared_runtime, "implemented");
    assert.equal(matrix.capabilities.current_user_sid_dacl, "blocked");
    assert.ok(
      matrix.conditions_to_unblock.includes(
        "windows_11_x64_target_machine_or_runner",
      ),
    );
  });

  // Fail-closed gate: the process exit code must agree with the matrix, not be
  // a constant 0. A release job that only checks step success must fail while
  // the platform is blocked. Removing the `process.exitCode = ...` line in the
  // script makes this test go red.
  it("exits non-zero while the Windows release gate is blocked", () => {
    const result = spawnSync(process.execPath, [script], { encoding: "utf8" });
    const matrix = JSON.parse(result.stdout);
    assert.equal(matrix.release_gate.status, "blocked");
    assert.notEqual(result.status, 0);
  });
});
