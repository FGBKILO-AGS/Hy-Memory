import type {
  GovernedMemory,
  MemoryKind,
  SdkMemoryRecord,
} from "@one-memory/contracts";

function kindOf(layer: string): MemoryKind {
  if (layer === "l6_schema") return "schema";
  if (layer === "l7_intention") return "intention";
  if (layer.startsWith("coding")) return "coding";
  if (layer.startsWith("working")) return "working";
  return "chat";
}

function stamp(seconds: number | null | undefined): string {
  return typeof seconds === "number" && Number.isFinite(seconds)
    ? new Date(seconds * 1_000).toISOString()
    : "";
}

/** An SDK memory in the shape the memory map draws.
 *
 * The map groups and labels by id, content, kind, status and tags only. It
 * never writes through this shape: opening a memory goes to the SDK library,
 * so the governed-only fields are inert placeholders here.
 */
export function sdkMapMemory(record: SdkMemoryRecord): GovernedMemory {
  const created = stamp(record.gmt_created ?? record.memory_at);
  return {
    memory_id: record.memory_id,
    user_id: record.user_id,
    kind: kindOf(record.layer),
    status:
      record.status === "active"
        ? "confirmed"
        : record.status === "superseded"
          ? "superseded"
          : "deleted",
    content: record.content,
    scopes: [],
    evidence_ids: [],
    origin: "extracted",
    tags: [...record.tags],
    supersedes: [...record.supersedes],
    superseded_by: [...record.superseded_by],
    revision: 1,
    created_at: created,
    updated_at: created,
    deleted_at: null,
  };
}
