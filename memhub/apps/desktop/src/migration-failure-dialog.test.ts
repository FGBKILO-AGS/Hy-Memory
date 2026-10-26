import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  MIGRATION_ISSUE_URL,
  type MigrationDialogDependencies,
  migrationFailureCode,
  migrationIssueUrl,
  migrationLogPath,
  showMigrationFailureDialog,
} from "./migration-failure-dialog";

async function home(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), "memhub-migration-dialog-"));
}

function dependencies(choice: number) {
  return {
    showMessageBoxSync: vi.fn(
      (
        _options: Parameters<
          MigrationDialogDependencies["showMessageBoxSync"]
        >[0],
      ) => choice,
    ),
    showItemInFolder: vi.fn(),
    openExternal: vi.fn(async () => undefined),
  };
}

describe("migration failure dialog", () => {
  it("names the code, reveals the log and opens a prefilled GitHub issue", async () => {
    const dataDir = await home();
    const deps = dependencies(0);

    const handled = showMigrationFailureDialog({
      error: new Error("sdk_migration_preflight_failed"),
      dataDir,
      locale: "zh-CN",
      dependencies: deps,
    });

    expect(handled).toBe(true);
    const options = deps.showMessageBoxSync.mock.calls[0]?.[0];
    expect(options).toMatchObject({
      type: "error",
      message: "MemHub 升级时数据迁移没有完成。",
      buttons: ["打开日志", "退出"],
      defaultId: 0,
      cancelId: 1,
    });
    expect(options?.detail).toContain(
      "为保护你的数据，MemHub 不会启动。请把迁移日志附在 GitHub 反馈里发给我们。（错误码：sdk_migration_preflight_failed）",
    );
    expect(options?.detail).toContain("sdk-migration.log");
    expect(deps.showItemInFolder).toHaveBeenCalledWith(
      migrationLogPath(dataDir),
    );
    expect(deps.openExternal).toHaveBeenCalledWith(
      migrationIssueUrl("sdk_migration_preflight_failed"),
    );
    expect(migrationIssueUrl("sdk_migration_preflight_failed")).toMatch(
      new RegExp(
        `^${MIGRATION_ISSUE_URL.replaceAll(".", "\\.")}\\?title=.*sdk_migration_preflight_failed$`,
      ),
    );
    // The file the dialog points at exists and ends with this failure.
    const last = (await readFile(migrationLogPath(dataDir), "utf8"))
      .trim()
      .split("\n")
      .at(-1);
    expect(JSON.parse(last ?? "{}")).toMatchObject({
      writer: "desktop",
      event: "desktop.migration_failure_shown",
      code: "sdk_migration_preflight_failed",
    });
  });

  it("quits without opening anything when the person chooses quit", async () => {
    const deps = dependencies(1);
    expect(
      showMigrationFailureDialog({
        error: new Error("sdk_migration_required"),
        dataDir: await home(),
        locale: "en-US",
        dependencies: deps,
      }),
    ).toBe(true);
    expect(deps.showMessageBoxSync.mock.calls[0]?.[0]?.buttons).toEqual([
      "Open log",
      "Quit",
    ]);
    expect(deps.showItemInFolder).not.toHaveBeenCalled();
    expect(deps.openExternal).not.toHaveBeenCalled();
  });

  it("uses the block note the service left when Desktop only saw 'not ready'", async () => {
    const dataDir = await home();
    await writeFile(
      path.join(dataDir, "sdk-migration-blocked.json"),
      JSON.stringify({
        schema_version: 1,
        code: "sdk_migration_model_profile_required",
        caller: "runtime-service",
        at: "2026-10-26T00:00:00.000Z",
      }),
    );
    expect(
      migrationFailureCode(new Error("runtime_service_not_ready"), dataDir),
    ).toBe("sdk_migration_model_profile_required");
    // A different, specific failure is not relabelled as a migration.
    expect(
      migrationFailureCode(new Error("installed_version_conflict"), dataDir),
    ).toBeNull();
  });

  it("leaves other startup failures to the generic error box", async () => {
    const deps = dependencies(0);
    expect(
      showMigrationFailureDialog({
        error: new Error("installed_version_conflict"),
        dataDir: await home(),
        locale: "zh-CN",
        dependencies: deps,
      }),
    ).toBe(false);
    expect(deps.showMessageBoxSync).not.toHaveBeenCalled();
  });
});
