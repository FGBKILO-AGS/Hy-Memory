import type {
  GovernedMemory,
  MemoryDetail,
  MemoryKind,
  MemoryLibraryPage,
  SdkMemoryPage,
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

const EARLIEST_PLAUSIBLE = Date.UTC(2000, 0, 1) / 1_000;

/** A conversation time that can be true: not before 2000, not tomorrow or later.
 *
 * A historical Cursor bug stored a file hash as the message time (years like
 * 2254). Such a value is not shown as when the conversation happened; the
 * import time, or "time unknown", says what is actually known.
 */
function plausible(seconds: number | null | undefined): number | null {
  return typeof seconds === "number" &&
    Number.isFinite(seconds) &&
    seconds >= EARLIEST_PLAUSIBLE &&
    seconds <= Date.now() / 1_000 + 86_400
    ? seconds
    : null;
}

/** The SDK writes "default" when no source was given; keep it as that bucket. */
function sourceOf(record: SdkMemoryRecord): string {
  return record.source || "default";
}

/** An SDK memory in the shape the memory map draws.
 *
 * The map groups and labels by id, content, kind, status and tags only, so the
 * governed-only fields are inert placeholders here. Opening, editing and
 * deleting still go through the SDK (v2) endpoints — `sdkDetail()` below wraps
 * the same projection for the map's inspector, which is why the inspector can
 * stay where the design put it instead of sending the reader to another view.
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
    // GovernedMemory still requires a revision for the shared map model. SDK
    // has no revision contract; the inspector recognises sdk_status and never
    // renders this placeholder.
    revision: 0,
    created_at: created,
    updated_at: created,
    deleted_at: null,
  };
}

/** One SDK memory as the map's inspector reads it.
 *
 * The inspector was built for a governed detail. SDK-native memory has no
 * product revision or "original evidence" section; source_raw_ids remain an
 * internal SDK provenance relation and are deliberately not projected as UI
 * evidence.
 */
export function sdkDetail(record: SdkMemoryRecord): MemoryDetail {
  return {
    memory: sdkMapMemory(record),
    evidence: [],
    history: [],
    index_sync: null,
    sdk_status: record.status,
    source_id: sourceOf(record),
    sdk_memory_at: stamp(plausible(record.memory_at)) || null,
    sdk_created_at: stamp(record.gmt_created) || null,
  };
}

/** One SDK page as the map's left list reads it.
 *
 * The list asks a library page for its rows, its total and — per row — when
 * the cited conversation happened. The SDK page carries the row itself, so the
 * time is taken from the memory: `memory_at` is the moment the conversation
 * happened, `gmt_created` only the moment it was imported. Naming which one it
 * is keeps "对话于" from silently becoming "沉淀于".
 */
export function sdkLibraryPage(page: SdkMemoryPage): MemoryLibraryPage {
  const sourceTime: Record<
    string,
    { at: string; basis: "conversation" | "import" }
  > = {};
  const sourceIds: Record<string, string[]> = {};
  for (const record of page.memories) {
    sourceIds[record.memory_id] = [sourceOf(record)];
    const conversation = plausible(record.memory_at);
    const at = conversation ?? plausible(record.gmt_created);
    if (at === null) continue;
    sourceTime[record.memory_id] = {
      at: new Date(at * 1_000).toISOString(),
      basis: conversation !== null ? "conversation" : "import",
    };
  }
  return {
    items: page.memories.map(sdkMapMemory),
    total: page.total,
    offset: page.offset,
    limit: page.limit,
    source_ids_by_memory: sourceIds,
    source_time_by_memory: sourceTime,
    search_window: page.search_window ?? null,
  };
}
