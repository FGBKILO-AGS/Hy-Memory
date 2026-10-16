import type { CoreReader } from "./core-client.js";
import type { CoreOperations } from "./core-operations.js";
import { PRODUCT } from "./version.js";

export type ModelContentPermissionDocument =
  | {
      schema_version: 1;
      product: string;
      state: "ok";
      allowed: boolean;
    }
  | {
      schema_version: 1;
      product: string;
      state: "rejected" | "unknown";
      code: string;
    };

export async function readModelContentPermission(
  reader: CoreReader,
): Promise<ModelContentPermissionDocument> {
  const outcome = await reader<unknown>("/api/v1/model-content-permission");
  if (outcome.state !== "ok") {
    return failure("unknown", outcome.code);
  }
  return permissionState(
    outcome.data,
    "model_content_permission_response_invalid",
  );
}

export async function setModelContentPermission(
  operations: CoreOperations,
  allowed: boolean,
): Promise<ModelContentPermissionDocument> {
  const outcome = await operations.post<unknown>(
    "/api/v1/model-content-permission",
    { allowed },
  );
  if (outcome.state === "rejected") {
    return failure("rejected", outcome.code);
  }
  if (outcome.state === "unknown") {
    return failure("unknown", outcome.code);
  }
  const document = permissionState(
    outcome.data,
    "model_content_permission_response_invalid",
  );
  return document.state === "ok" && document.allowed !== allowed
    ? failure("unknown", "model_content_permission_state_mismatch")
    : document;
}

export function modelContentPermissionExitCode(
  document: ModelContentPermissionDocument,
): number {
  return document.state === "ok" ? 0 : 1;
}

export function renderModelContentPermissionText(
  document: ModelContentPermissionDocument,
): string {
  if (document.state === "ok") {
    return `模型内容外发：${document.allowed ? "允许" : "禁止"}\n`;
  }
  return `模型内容权限未更新：${document.state}（${document.code}）\n`;
}

function permissionState(
  value: unknown,
  invalidCode: string,
): ModelContentPermissionDocument {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    Object.keys(value).length !== 1 ||
    typeof (value as Record<string, unknown>).allowed !== "boolean"
  ) {
    return failure("unknown", invalidCode);
  }
  return {
    schema_version: 1,
    product: PRODUCT,
    state: "ok",
    allowed: (value as { allowed: boolean }).allowed,
  };
}

function failure(
  state: "rejected" | "unknown",
  code: string,
): ModelContentPermissionDocument {
  return { schema_version: 1, product: PRODUCT, state, code };
}
