// @vitest-environment jsdom
import type {
  GovernedMemory,
  MemoryGraphResponse,
  SdkMemoryRecord,
} from "@one-memory/contracts";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MemorySpace } from "./MemorySpace";
import { sdkMapMemory } from "./sdk-memory-map";

const api = vi.hoisted(() => ({
  sdkLibrary: vi.fn(),
  sdkGraph: vi.fn(),
  sdkMemory: vi.fn(),
  updateSdkMemory: vi.fn(),
  deleteSdkMemory: vi.fn(),
}));
// Only the SDK (v2) reads and writes exist: a call to anything else fails the
// test with "is not a function" instead of passing silently.
vi.mock("./api", () => ({
  getSdkMemoryLibrary: api.sdkLibrary,
  getSdkMemoryGraph: api.sdkGraph,
  getSdkMemory: api.sdkMemory,
  updateSdkMemory: api.updateSdkMemory,
  deleteSdkMemory: api.deleteSdkMemory,
  getMemoryProcessing: async () => null,
  getModelProfiles: async () => [],
  apiErrorCode: () => null,
}));
vi.mock("./GraphWidthPrompt", () => ({ GraphWidthPrompt: () => null }));
vi.mock("./MemoryNoteComposer", () => ({ MemoryNoteComposer: () => null }));
vi.mock("./MemoryMap", () => ({
  MemoryMap: ({
    hidden,
    memories,
    graph,
    loaded,
    busy,
    detail,
    onOpenMemory,
    onSave,
    onDelete,
    libraryView,
  }: {
    hidden: boolean;
    memories: GovernedMemory[];
    graph: MemoryGraphResponse | null;
    loaded: boolean;
    busy: boolean;
    detail: { memory: { memory_id: string } } | null;
    onOpenMemory(id: string): void;
    onSave(event: { preventDefault(): void }): void;
    onDelete(): void;
    libraryView?: {
      page: { source_ids_by_memory?: Record<string, string[]> } | null;
      unattributedSource?: string;
      onSource(value: string): void;
      resetPage(): void;
    };
  }) => (
    <div hidden={hidden} data-testid="map">
      <span data-testid="map-memories">
        {memories.map((m) => `${m.memory_id}:${m.status}`).join(",")}
      </span>
      <span data-testid="map-nodes">{graph?.nodes.length ?? "none"}</span>
      <span data-testid="map-loaded">{String(loaded)}</span>
      <span data-testid="map-sources">
        {memories
          .map((m) =>
            (libraryView?.page?.source_ids_by_memory?.[m.memory_id] ?? []).join(
              "+",
            ),
          )
          .join(",")}
      </span>
      <span data-testid="map-unattributed">
        {libraryView?.unattributedSource ?? ""}
      </span>
      <span data-testid="map-busy">{String(busy)}</span>
      <span data-testid="map-detail">{detail?.memory.memory_id ?? ""}</span>
      <button type="button" onClick={() => onOpenMemory("a")}>
        open a
      </button>
      <button type="button" onClick={() => onSave({ preventDefault() {} })}>
        save
      </button>
      <button type="button" onClick={onDelete}>
        delete
      </button>
      <button
        type="button"
        onClick={() => {
          libraryView?.onSource("workbuddy");
          libraryView?.resetPage();
        }}
      >
        pick workbuddy
      </button>
    </div>
  ),
}));
vi.mock("./SdkMemoryLibrary", () => ({
  SdkMemoryLibrary: ({
    focusedMemoryId,
  }: {
    focusedMemoryId: string | null;
  }) => <div data-testid="sdk-library">{focusedMemoryId ?? ""}</div>,
}));

const record = (memory_id: string): SdkMemoryRecord => ({
  memory_id,
  content: `memory ${memory_id}`,
  layer: "l2_fact",
  status: "active",
  is_latest: true,
  supersedes: [],
  superseded_by: [],
  tags: ["工作"],
  owner: null,
  source_raw_ids: [],
  user_id: "local",
  agent_id: "all-apps",
  session_id: "s1",
  source: "cursor",
  memory_at: null,
  gmt_created: 1_761_000_000,
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function sdkAuthority() {
  api.sdkLibrary.mockResolvedValue({
    memories: [record("a"), record("b")],
    total: 3018,
    limit: 300,
    offset: 0,
  });
  api.sdkGraph.mockResolvedValue({
    total_memories: 3018,
    projected_memories: 2,
    nodes: [
      { node_id: "memory:a", node_type: "memory", label: "a", memory_id: "a" },
      { node_id: "memory:b", node_type: "memory", label: "b", memory_id: "b" },
      { node_id: "source:cursor", node_type: "source", label: "Cursor" },
    ],
    edges: [],
    hy_graph_available: true,
  });
}

it("draws the map from SDK memories on SDK authority, without legacy reads", async () => {
  sdkAuthority();

  render(<MemorySpace simplified active />);

  await waitFor(() =>
    expect(screen.getByTestId("map-loaded").textContent).toBe("true"),
  );
  expect(screen.getByTestId("map").hidden).toBe(false);
  expect(screen.getByTestId("map-memories").textContent).toBe(
    "a:confirmed,b:confirmed",
  );
  expect(screen.getByTestId("map-nodes").textContent).toBe("3");
  expect(await screen.findByText(/3,018/)).toBeTruthy();
  expect(api.sdkLibrary).toHaveBeenCalledWith(
    {
      statuses: ["active"],
      source: undefined,
      query: undefined,
      offset: 0,
      limit: 100,
    },
    expect.any(AbortSignal),
  );
  expect(api.sdkGraph).toHaveBeenCalledWith(
    { source: undefined, offset: 0, limit: 100 },
    expect.any(AbortSignal),
  );
});

it("keeps rows, sources, count and graph on the one filtered page", async () => {
  sdkAuthority();
  render(<MemorySpace simplified active />);
  await waitFor(() =>
    expect(screen.getByTestId("map-loaded").textContent).toBe("true"),
  );
  api.sdkLibrary.mockResolvedValue({
    memories: [
      { ...record("w1"), source: "workbuddy" },
      { ...record("w2"), source: "default" },
    ],
    total: 566,
    limit: 100,
    offset: 0,
  });

  fireEvent.click(screen.getByText("pick workbuddy"));

  await waitFor(() =>
    expect(screen.getByTestId("map-memories").textContent).toBe(
      "w1:confirmed,w2:confirmed",
    ),
  );
  expect(screen.getByTestId("map-sources").textContent).toBe(
    "workbuddy,default",
  );
  expect(screen.getByTestId("map-unattributed").textContent).toBe("default");
  expect(api.sdkLibrary).toHaveBeenLastCalledWith(
    expect.objectContaining({ source: "workbuddy", offset: 0 }),
    expect.any(AbortSignal),
  );
  expect(api.sdkGraph).toHaveBeenLastCalledWith(
    { source: "workbuddy", offset: 0, limit: 100 },
    expect.any(AbortSignal),
  );
  expect(await screen.findByText(/566/)).toBeTruthy();
});

it("opens an SDK memory in place instead of switching away from the map", async () => {
  sdkAuthority();
  api.sdkMemory.mockResolvedValue(record("a"));

  render(<MemorySpace simplified active />);
  await waitFor(() =>
    expect(screen.getByTestId("map-loaded").textContent).toBe("true"),
  );
  fireEvent.click(screen.getByText("open a"));

  await waitFor(() => expect(api.sdkMemory).toHaveBeenCalledWith("a"));
  // The inspector stays on the map; the reader is never moved to a second list.
  expect(screen.getByTestId("map").hidden).toBe(false);
  expect(screen.getByTestId("sdk-library").textContent).toBe("");
});

it("re-reads the page on the next tick when Core is not ready yet", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    sdkAuthority();
    api.sdkLibrary.mockRejectedValueOnce(new Error("core_not_ready"));

    render(<MemorySpace simplified active />);
    await waitFor(() => expect(api.sdkLibrary).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("map-loaded").textContent).toBe("false");
    await vi.advanceTimersByTimeAsync(30_000);

    await waitFor(() =>
      expect(screen.getByTestId("map-loaded").textContent).toBe("true"),
    );
  } finally {
    vi.useRealTimers();
  }
});

it("saves and deletes an opened memory through v2 without a stale-read block", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    sdkAuthority();
    api.sdkMemory.mockResolvedValue(record("a"));
    api.updateSdkMemory.mockResolvedValue({
      ...record("a"),
      content: "rewritten",
    });
    api.deleteSdkMemory.mockResolvedValue(undefined);

    render(<MemorySpace simplified active />);
    await waitFor(() =>
      expect(screen.getByTestId("map-loaded").textContent).toBe("true"),
    );
    fireEvent.click(screen.getByText("open a"));
    await waitFor(() =>
      expect(screen.getByTestId("map-detail").textContent).toBe("a"),
    );
    // The governed detail used to be re-verified every 10s through v1; on SDK
    // that read was refused and left the detail blocked as busy.
    await vi.advanceTimersByTimeAsync(25_000);
    expect(screen.getByTestId("map-busy").textContent).toBe("false");

    fireEvent.click(screen.getByText("save"));
    await waitFor(() =>
      expect(api.updateSdkMemory).toHaveBeenCalledWith(
        "a",
        "memory a",
        expect.any(String),
      ),
    );
    await waitFor(() =>
      expect(screen.getByTestId("map-busy").textContent).toBe("false"),
    );

    fireEvent.click(screen.getByText("delete"));
    fireEvent.click(await screen.findByRole("button", { name: "确认删除" }));
    await waitFor(() =>
      expect(api.deleteSdkMemory).toHaveBeenCalledWith("a", expect.any(String)),
    );
    await waitFor(() =>
      expect(screen.getByTestId("map-detail").textContent).toBe(""),
    );
  } finally {
    vi.useRealTimers();
  }
});

it("projects SDK records into the fields the map groups by", () => {
  const coding = sdkMapMemory({ ...record("c"), layer: "coding_experience" });
  expect(coding).toMatchObject({
    memory_id: "c",
    kind: "coding",
    status: "confirmed",
    tags: ["工作"],
    deleted_at: null,
  });
  expect(sdkMapMemory({ ...record("s"), layer: "l6_schema" }).kind).toBe(
    "schema",
  );
  expect(sdkMapMemory({ ...record("x"), status: "superseded" }).status).toBe(
    "superseded",
  );
});

const node = (
  node_id: string,
  node_type: MemoryGraphResponse["nodes"][number]["node_type"],
  extra: Partial<MemoryGraphResponse["nodes"][number]> = {},
): MemoryGraphResponse["nodes"][number] => ({
  node_id,
  node_type,
  label: node_id,
  memory_id: null,
  source_id: null,
  kind: null,
  status: null,
  confidence: null,
  ...extra,
});
const provenance = (
  from: string,
  to: string,
): MemoryGraphResponse["edges"][number] => ({
  edge_id: `${from}->${to}`,
  source_node_id: from,
  target_node_id: to,
  relation: "evidence_source",
  weight: null,
  origin: "provenance",
});

it("groups and draws SDK memories by source with the real map model", async () => {
  const { mapGroups, projectMap } = await import("./memory-map-model");
  const memories = [record("a"), record("b")].map(sdkMapMemory);
  const graph: MemoryGraphResponse = {
    total_memories: 2,
    projected_memories: 2,
    nodes: [
      node("memory:a", "memory", { memory_id: "a" }),
      node("memory:b", "memory", { memory_id: "b" }),
      node("source:cursor", "source", { source_id: "cursor", label: "Cursor" }),
      node("hy:t1", "topic"),
    ],
    edges: [
      provenance("source:cursor", "memory:a"),
      provenance("source:cursor", "memory:b"),
    ],
    hy_graph_available: true,
  };

  const groups = mapGroups(memories, graph, "source");
  expect(groups.map((group) => [group.key, group.memoryIds])).toEqual([
    ["source:cursor", ["a", "b"]],
  ]);
  const drawn = projectMap(graph, memories, "source", ["a", "b"]);
  expect(
    drawn.nodes
      .filter((node) => node.node_type === "memory")
      .map((n) => n.memory_id),
  ).toEqual(["a", "b"]);
  expect(drawn.edges.length).toBeGreaterThan(0);
});
