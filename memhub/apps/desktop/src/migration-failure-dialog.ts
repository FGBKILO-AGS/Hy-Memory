import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";

/**
 * What a person sees when an upgrade did not reach SDK authority.
 *
 * MemHub does not run on a home whose migration did not finish: one data shape
 * per release, no legacy fallback. The app therefore stops, and this dialog is
 * the only thing between the failure and a useful report — it names the code
 * and hands the person the one file support needs (`sdk-migration.log`).
 */

export const MIGRATION_ISSUE_URL =
  "https://github.com/Tencent-Hunyuan/Hy-Memory/issues/new";

const CODE = /^[a-z][a-z0-9_]{0,95}$/;
/** Generic startup codes that hide a migration block recorded by the service. */
const MASKING_CODES = new Set([
  "desktop_start_failed",
  "runtime_service_not_ready",
]);

export function migrationLogPath(dataDir: string): string {
  return path.join(dataDir, "logs", "sdk-migration.log");
}

function readBlockCode(dataDir: string): string | null {
  try {
    const value = JSON.parse(
      readFileSync(path.join(dataDir, "sdk-migration-blocked.json"), "utf8"),
    ) as { code?: unknown };
    return typeof value.code === "string" && CODE.test(value.code)
      ? value.code
      : null;
  } catch {
    return null;
  }
}

/**
 * The migration code behind a startup failure, or null when it was not one.
 *
 * Desktop's own migration throws the code directly. After a login launch the
 * service migrates instead, and Desktop only sees "not ready": the block note
 * the service left says why.
 */
export function migrationFailureCode(
  error: unknown,
  dataDir: string | null,
): string | null {
  const message = error instanceof Error ? error.message : "";
  if (CODE.test(message) && message.startsWith("sdk_migration_")) {
    return message;
  }
  if (dataDir === null) return null;
  const blocked = readBlockCode(dataDir);
  if (blocked === null) return null;
  return MASKING_CODES.has(message) ||
    !CODE.test(message) ||
    message.startsWith("core_")
    ? blocked
    : null;
}

export function migrationIssueUrl(code: string): string {
  const title = `[升级迁移失败 / migration failed] ${code}`;
  return `${MIGRATION_ISSUE_URL}?${new URLSearchParams({ title }).toString()}`;
}

export interface MigrationDialogText {
  title: string;
  message: string;
  detail: string;
  openLog: string;
  quit: string;
}

export function migrationDialogText(
  code: string,
  logPath: string,
  locale: string,
): MigrationDialogText {
  if (locale.toLowerCase().startsWith("en")) {
    return {
      title: "MemHub",
      message: "The data migration for this MemHub upgrade did not finish.",
      detail:
        "To protect your data, MemHub will not start. Please attach the migration log to a GitHub issue and send it to us. " +
        `(Error code: ${code})\n\nLog: ${logPath}`,
      openLog: "Open log",
      quit: "Quit",
    };
  }
  return {
    title: "MemHub",
    message: "MemHub 升级时数据迁移没有完成。",
    detail:
      "为保护你的数据，MemHub 不会启动。请把迁移日志附在 GitHub 反馈里发给我们。" +
      `（错误码：${code}）\n\n日志：${logPath}`,
    openLog: "打开日志",
    quit: "退出",
  };
}

export interface MigrationDialogDependencies {
  showMessageBoxSync(options: {
    type: "error";
    title: string;
    message: string;
    detail: string;
    buttons: string[];
    defaultId: number;
    cancelId: number;
    noLink: boolean;
  }): number;
  showItemInFolder(target: string): void;
  openExternal(url: string): Promise<void> | void;
}

/** Make sure the file the dialog points at exists and ends with this failure. */
function appendFinalLine(dataDir: string, code: string): void {
  try {
    const target = migrationLogPath(dataDir);
    mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
    appendFileSync(
      target,
      `${JSON.stringify({
        schema_version: 1,
        ts: new Date().toISOString(),
        pid: process.pid,
        writer: "desktop",
        event: "desktop.migration_failure_shown",
        outcome: "refused",
        code,
        platform: process.platform,
      })}\n`,
      { encoding: "utf8", mode: 0o600 },
    );
  } catch {
    // The dialog still names the code even if the log cannot be written.
  }
}

/**
 * Show the migration failure dialog. Returns true when it handled the error;
 * the caller then quits without showing any window.
 */
export function showMigrationFailureDialog(options: {
  error: unknown;
  dataDir: string | null;
  locale: string;
  dependencies: MigrationDialogDependencies;
}): boolean {
  const code = migrationFailureCode(options.error, options.dataDir);
  if (code === null || options.dataDir === null) return false;
  appendFinalLine(options.dataDir, code);
  const logPath = migrationLogPath(options.dataDir);
  const text = migrationDialogText(code, logPath, options.locale);
  const choice = options.dependencies.showMessageBoxSync({
    type: "error",
    title: text.title,
    message: text.message,
    detail: text.detail,
    buttons: [text.openLog, text.quit],
    defaultId: 0,
    cancelId: 1,
    noLink: true,
  });
  if (choice === 0) {
    options.dependencies.showItemInFolder(logPath);
    void Promise.resolve(
      options.dependencies.openExternal(migrationIssueUrl(code)),
    ).catch(() => undefined);
  }
  return true;
}
