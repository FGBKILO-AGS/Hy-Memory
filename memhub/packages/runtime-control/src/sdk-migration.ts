import { spawn } from "node:child_process";
import { withRuntimeMaintenanceFence } from "./runtime-maintenance.js";
import {
  clearSdkMigrationBlock,
  migrationErrorCode,
  recordSdkMigrationBlock,
  type SdkMigrationCaller,
  type SdkMigrationLogEntry,
  writeSdkMigrationLog,
} from "./sdk-migration-log.js";

const MAX_OUTPUT_BYTES = 1024 * 1024;

export interface SdkMigrationResult {
  /** `complete`: already on SDK authority; leftovers were repaired. */
  state: "not_required" | "cutover" | "complete" | "deferred";
  code?: string;
}

export interface SdkMigrationCommandResult {
  exitCode: number;
  document: Record<string, unknown>;
  /** Size of Core's stderr. Its text can carry content, so only the size is logged. */
  stderrBytes?: number;
}

export interface CoreCommand {
  command: string;
  args?: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
}

export interface SdkMigrationOptions {
  dataDir: string;
  core: CoreCommand;
  runtimeBuild: string;
  execute?: (
    launch: CoreCommand,
    args: string[],
    dataDir: string,
    onProgress?: (step: string) => void,
  ) => Promise<SdkMigrationCommandResult>;
  fence?: <T>(dataDir: string, action: () => Promise<T>) => Promise<T>;
  /** Called with `bundle`, `prepare`, `cutover`, `cleanup` as Core reaches them. */
  onProgress?: (step: string) => void;
  /** Who is migrating; stamped on every line of the migration log. */
  caller?: SdkMigrationCaller;
}

/**
 * Bring a home to SDK memory authority before a Runtime serves it.
 *
 * Core's `sdk-migrate` does the whole migration and resumes after any failure;
 * the caller (Desktop startup, CLI install/upgrade, the Runtime service) only
 * keeps the Runtime stopped, and its service from restarting, while it runs.
 *
 * Every decision is appended to `<dataDir>/logs/sdk-migration.log`, and Core
 * writes its own phase and failure lines to the same file, so one attachment
 * explains an upgrade that did not finish. A deferred result leaves a durable
 * block note that `memhub status` and Desktop read; a successful one clears it.
 */
export async function runSdkMigration(
  options: SdkMigrationOptions,
): Promise<SdkMigrationResult> {
  const caller = options.caller ?? "desktop";
  const log = (entry: Omit<SdkMigrationLogEntry, "writer" | "runtimeBuild">) =>
    writeSdkMigrationLog(options.dataDir, {
      writer: caller,
      runtimeBuild: options.runtimeBuild,
      ...entry,
    });
  const started = Date.now();
  const core: CoreCommand = {
    ...options.core,
    env: {
      ...options.core.env,
      // Core stamps these on its own lines in the shared log.
      ONE_MEMORY_MIGRATION_CALLER: caller,
      ONE_MEMORY_MIGRATION_RUNTIME_BUILD: options.runtimeBuild,
    },
  };
  const finish = async (
    result: SdkMigrationResult,
    details: Record<string, unknown> = {},
  ): Promise<SdkMigrationResult> => {
    await log({
      event: "migration.finished",
      outcome: result.state === "deferred" ? "deferred" : "succeeded",
      ...(result.code === undefined ? {} : { code: result.code }),
      details: {
        state: result.state,
        duration_ms: Date.now() - started,
        ...details,
      },
    });
    if (result.state === "deferred") {
      await recordSdkMigrationBlock(
        options.dataDir,
        result.code ?? "sdk_migration_incomplete",
        caller,
      );
    } else {
      await clearSdkMigrationBlock(options.dataDir);
    }
    return result;
  };
  const execute = options.execute ?? executeCoreCommand;
  const fence = options.fence ?? withRuntimeMaintenanceFence;
  await log({ event: "migration.invoked", outcome: "started" });
  let statusFailure = "";
  const status = await execute(
    core,
    ["sdk-migration-status", "--json"],
    options.dataDir,
  ).catch((error: unknown) => {
    statusFailure = migrationErrorCode(error, "sdk_migration_status_failed");
    return null;
  });
  await log({
    event: "migration.status_probed",
    outcome: status !== null && status.exitCode === 0 ? "succeeded" : "failed",
    details: {
      exit_code: status?.exitCode ?? null,
      stderr_bytes: status?.stderrBytes ?? null,
      authority: status?.document.authority ?? null,
      state: status?.document.state ?? null,
      ...(statusFailure ? { reason: statusFailure } : {}),
    },
  });
  if (status === null || status.exitCode !== 0) {
    // An unreadable status still must not leave a pre-upgrade Runtime serving
    // the home: cycle the service so launchd brings back the gated `current`.
    await fence(options.dataDir, async () => undefined).catch(() => undefined);
    return finish({
      state: "deferred",
      code: "sdk_migration_status_unavailable",
    });
  }
  if (status.document.state !== "required") {
    return finish({ state: "not_required" });
  }
  return fence(options.dataDir, async () => {
    await log({ event: "migration.started", outcome: "started" });
    let migrated: SdkMigrationCommandResult;
    try {
      migrated = await execute(
        core,
        [
          "sdk-migrate",
          "--runtime-build",
          options.runtimeBuild,
          "--yes",
          "--json",
        ],
        options.dataDir,
        (step) => {
          void log({
            event: "migration.progress",
            outcome: "started",
            details: { step },
          });
          options.onProgress?.(step);
        },
      );
    } catch (error) {
      return finish(
        {
          state: "deferred",
          code: migrationErrorCode(error, "sdk_migration_response_invalid"),
        },
        { stage: "sdk_migrate_spawn" },
      );
    }
    const childDetails = {
      exit_code: migrated.exitCode,
      stderr_bytes: migrated.stderrBytes ?? null,
      document_state: migrated.document.state ?? null,
      document_code: migrated.document.code ?? null,
      migration_id: migrated.document.migration_id ?? null,
    };
    if (migrated.exitCode !== 0)
      return finish(deferred(migrated), childDetails);
    const state = migrated.document.state;
    if (state === "cutover" || state === "complete") {
      return finish({ state }, childDetails);
    }
    if (state === "not_required") {
      return finish({ state: "not_required" }, childDetails);
    }
    return finish(deferred(migrated), childDetails);
  });
}

function deferred(result: SdkMigrationCommandResult): SdkMigrationResult {
  return {
    state: "deferred",
    code:
      typeof result.document.code === "string"
        ? result.document.code
        : "sdk_migration_incomplete",
  };
}

async function executeCoreCommand(
  launch: CoreCommand,
  args: string[],
  dataDir: string,
  onProgress?: (step: string) => void,
): Promise<SdkMigrationCommandResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(launch.command, [...(launch.args ?? []), ...args], {
      cwd: launch.cwd,
      env: { ...process.env, ...launch.env, ONE_MEMORY_DATA_DIR: dataDir },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let reported = 0;
    let stderrBytes = 0;
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
      const complete = stdout.lastIndexOf("\n");
      if (onProgress && complete > reported) {
        for (const line of stdout.slice(reported, complete).split("\n")) {
          const step = progressStep(line);
          if (step !== null) onProgress(step);
        }
        reported = complete + 1;
      }
      if (Buffer.byteLength(stdout) > MAX_OUTPUT_BYTES) {
        child.kill("SIGKILL");
        reject(new Error("sdk_migration_output_too_large"));
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderrBytes += chunk.length;
      if (stderrBytes > MAX_OUTPUT_BYTES) {
        child.kill("SIGKILL");
        reject(new Error("sdk_migration_error_output_too_large"));
      }
    });
    child.once("error", reject);
    child.once("close", (code) => {
      try {
        const line = stdout.trim().split("\n").at(-1) ?? "{}";
        const document = JSON.parse(line) as unknown;
        if (
          !document ||
          typeof document !== "object" ||
          Array.isArray(document)
        ) {
          throw new Error("sdk_migration_response_invalid");
        }
        resolve({
          exitCode: code ?? 1,
          document: document as Record<string, unknown>,
          stderrBytes,
        });
      } catch {
        reject(new Error("sdk_migration_response_invalid"));
      }
    });
  });
}

function progressStep(line: string): string | null {
  try {
    const document = JSON.parse(line) as { state?: unknown; step?: unknown };
    return document.state === "running" && typeof document.step === "string"
      ? document.step
      : null;
  } catch {
    return null;
  }
}
