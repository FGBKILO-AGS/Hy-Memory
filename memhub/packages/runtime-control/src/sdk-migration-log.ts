import {
  appendFile,
  chmod,
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";

/**
 * The one log support asks for when an upgrade did not reach SDK authority.
 *
 * `<dataDir>/logs/sdk-migration.log` is JSON Lines shared with Core's
 * `one_memory/sdk_migration_log.py`, which writes the same line shape. It never
 * carries user content: detail values are numbers, booleans and short
 * identifier-shaped strings; anything else is dropped. Writing never throws —
 * logging must not be the reason a migration or a startup fails.
 */

export type SdkMigrationCaller =
  | "desktop"
  | "desktop-background"
  | "cli-install"
  | "cli-upgrade"
  | "runtime-service";

export const SDK_MIGRATION_LOG_NAME = "sdk-migration.log";
const MAX_BYTES = 4 * 1024 * 1024;
const SAFE_TEXT = /^[A-Za-z0-9_.:/@+-]{0,160}$/;
const MAX_ITEMS = 32;

export function sdkMigrationLogPath(dataDir: string): string {
  return path.join(dataDir, "logs", SDK_MIGRATION_LOG_NAME);
}

export interface SdkMigrationLogEntry {
  writer: SdkMigrationCaller;
  event: string;
  outcome: "started" | "succeeded" | "failed" | "deferred" | "refused";
  code?: string;
  migrationId?: string;
  runtimeBuild?: string;
  details?: Record<string, unknown>;
}

function safe(value: unknown, depth = 0): unknown {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") {
    return Number.isFinite(value) ? Math.round(value * 1000) / 1000 : null;
  }
  if (typeof value === "string") {
    return SAFE_TEXT.test(value) ? value : "<redacted>";
  }
  if (depth >= 4) return "<nested>";
  if (Array.isArray(value)) {
    const items = value
      .slice(0, MAX_ITEMS)
      .map((item) => safe(item, depth + 1));
    if (value.length > MAX_ITEMS) items.push(`<+${value.length - MAX_ITEMS}>`);
    return items;
  }
  if (typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value).slice(0, MAX_ITEMS)) {
      if (SAFE_TEXT.test(key)) result[key] = safe(item, depth + 1);
    }
    return result;
  }
  return "<unsupported>";
}

/** A stable code from an error, never its free-text message. */
export function migrationErrorCode(error: unknown, fallback: string): string {
  const message = error instanceof Error ? error.message : "";
  return /^[a-z][a-z0-9_]{0,95}$/.test(message) ? message : fallback;
}

export async function writeSdkMigrationLog(
  dataDir: string,
  entry: SdkMigrationLogEntry,
): Promise<void> {
  try {
    const line = safe({
      schema_version: 1,
      ts: new Date().toISOString(),
      pid: process.pid,
      writer: entry.writer,
      event: entry.event,
      outcome: entry.outcome,
      ...(entry.code === undefined ? {} : { code: entry.code }),
      ...(entry.migrationId === undefined
        ? {}
        : { migration_id: entry.migrationId }),
      ...(entry.runtimeBuild === undefined
        ? {}
        : { runtime_build: entry.runtimeBuild }),
      platform: process.platform,
      ...(entry.details === undefined ? {} : { details: entry.details }),
    });
    const target = sdkMigrationLogPath(dataDir);
    await mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
    const size = await stat(target).then(
      (info) => info.size,
      () => 0,
    );
    if (size > MAX_BYTES) await rename(target, `${target}.1`);
    await appendFile(target, `${JSON.stringify(line)}\n`, {
      encoding: "utf8",
      mode: 0o600,
    });
  } catch {
    // Never let the log decide whether a migration or a startup happens.
  }
}

/**
 * A durable note that this home is blocked on an unfinished migration.
 *
 * Written when a migration is deferred and removed once one succeeds, so a
 * separate process (`memhub status`, the Desktop dialog after a background
 * launch) reports the real cause instead of a generic "not running".
 */
export interface SdkMigrationBlock {
  code: string;
  caller: SdkMigrationCaller;
  at: string;
}

const BLOCK_NAME = "sdk-migration-blocked.json";

function blockPath(dataDir: string): string {
  return path.join(dataDir, BLOCK_NAME);
}

export async function recordSdkMigrationBlock(
  dataDir: string,
  code: string,
  caller: SdkMigrationCaller,
): Promise<void> {
  try {
    await mkdir(dataDir, { recursive: true, mode: 0o700 });
    const target = blockPath(dataDir);
    const temporary = `${target}.${process.pid}.tmp`;
    await writeFile(
      temporary,
      `${JSON.stringify({ schema_version: 1, code, caller, at: new Date().toISOString() })}\n`,
      { encoding: "utf8", mode: 0o600 },
    );
    await chmod(temporary, 0o600);
    await rename(temporary, target);
  } catch {
    // Best effort, like the log.
  }
}

export async function clearSdkMigrationBlock(dataDir: string): Promise<void> {
  await rm(blockPath(dataDir), { force: true }).catch(() => undefined);
}

export async function readSdkMigrationBlock(
  dataDir: string,
): Promise<SdkMigrationBlock | null> {
  try {
    const value = JSON.parse(
      await readFile(blockPath(dataDir), "utf8"),
    ) as Partial<SdkMigrationBlock>;
    if (
      typeof value.code !== "string" ||
      !/^[a-z][a-z0-9_]{0,95}$/.test(value.code) ||
      typeof value.caller !== "string" ||
      typeof value.at !== "string"
    ) {
      return null;
    }
    return value as SdkMigrationBlock;
  } catch {
    return null;
  }
}
