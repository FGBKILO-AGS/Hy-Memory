import type { SdkMemoryRecord } from "@one-memory/contracts";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { projectPagedSourceMap } from "./memory-map-model";
import { sdkDetail, sdkLibraryPage } from "./sdk-memory-map";

const NOW = Date.UTC(2026, 9, 25) / 1_000;

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(NOW * 1_000);
});
afterEach(() => vi.useRealTimers());

const record = (extra: Partial<SdkMemoryRecord>): SdkMemoryRecord => ({
  memory_id: "m",
  content: "memory",
  layer: "l2_fact",
  status: "active",
  is_latest: true,
  supersedes: [],
  superseded_by: [],
  tags: [],
  owner: null,
  source_raw_ids: [],
  user_id: "local",
  agent_id: "all-apps",
  session_id: "s1",
  source: "cursor",
  memory_at: null,
  gmt_created: NOW - 86_400,
  ...extra,
});

it("keeps the SDK's no-source bucket as a source the list and map can name", () => {
  const page = sdkLibraryPage({
    memories: [record({ memory_id: "d", source: "default" })],
    total: 1,
    limit: 100,
    offset: 0,
  });
  expect(page.source_ids_by_memory).toEqual({ d: ["default"] });
  expect(sdkDetail(record({ source: "default" })).source_id).toBe("default");

  const drawn = projectPagedSourceMap(null, page, [], {
    default: "未识别来源",
  });
  expect(
    drawn.nodes.find((node) => node.node_id === "source:default")?.label,
  ).toBe("未识别来源");
});

it("does not present an impossible conversation time as when it happened", () => {
  const future = Date.UTC(2254, 0, 13) / 1_000;
  const page = sdkLibraryPage({
    memories: [record({ memory_id: "f", memory_at: future })],
    total: 1,
    limit: 100,
    offset: 0,
  });
  expect(page.source_time_by_memory?.f).toEqual({
    at: new Date((NOW - 86_400) * 1_000).toISOString(),
    basis: "import",
  });
  expect(sdkDetail(record({ memory_at: future })).sdk_memory_at).toBeNull();

  const real = NOW - 3 * 86_400;
  expect(
    sdkLibraryPage({
      memories: [record({ memory_id: "r", memory_at: real })],
      total: 1,
      limit: 100,
      offset: 0,
    }).source_time_by_memory?.r.basis,
  ).toBe("conversation");
});

it("carries a full search window through to the list", () => {
  expect(
    sdkLibraryPage({
      memories: [],
      total: 0,
      limit: 100,
      offset: 0,
      search_window: 500,
    }).search_window,
  ).toBe(500);
});
