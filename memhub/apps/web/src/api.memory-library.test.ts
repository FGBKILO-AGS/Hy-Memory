import { beforeEach, describe, expect, it, vi } from "vitest";

const { get } = vi.hoisted(() => ({ get: vi.fn() }));
vi.mock("axios", () => ({
  default: {
    create: () => ({
      get,
      interceptors: { request: { use: vi.fn() } },
    }),
  },
}));

import { getHomeSummary, getSdkMemoryLibrary } from "./api";

describe("paged memory library client", () => {
  beforeEach(() => vi.clearAllMocks());

  it("sends SDK lifecycle filters as repeated status parameters", async () => {
    const page = { memories: [], total: 0, offset: 0, limit: 25 };
    get.mockResolvedValueOnce({ data: page });
    expect(
      await getSdkMemoryLibrary({
        statuses: ["active", "superseded"],
        source: "cursor",
        query: "release",
        offset: 0,
        limit: 25,
      }),
    ).toBe(page);
    expect(get.mock.calls[0][0]).toBe("/api/v2/memory-library");
    const params = get.mock.calls[0][1].params as URLSearchParams;
    expect(params.getAll("statuses")).toEqual(["active", "superseded"]);
    expect(params.get("limit")).toBe("25");
    expect(params.get("source")).toBe("cursor");
    expect(params.get("query")).toBe("release");
  });

  it("forwards cancellation for inactive or superseded views", async () => {
    const controller = new AbortController();
    get.mockResolvedValueOnce({
      data: { memories: [], total: 0, offset: 0, limit: 100 },
    });
    await getSdkMemoryLibrary({}, controller.signal);
    expect(get.mock.calls[0][1].signal).toBe(controller.signal);
  });

  it("does not turn a failed query into an empty library or fall back", async () => {
    const error = new Error("hy_memory_list_failed");
    get.mockRejectedValueOnce(error);
    await expect(getSdkMemoryLibrary()).rejects.toBe(error);
    expect(get).toHaveBeenCalledTimes(1);
  });
});

describe("home summary client", () => {
  beforeEach(() => vi.clearAllMocks());

  it("reads the SDK summary directly, with no authority request first", async () => {
    const summary = {
      schema_version: 2,
      memory_count: 3,
      latest: null,
      observed_at: "2026-10-26T00:00:00Z",
    };
    get.mockResolvedValueOnce({ data: summary });
    expect(await getHomeSummary()).toBe(summary);
    expect(get).toHaveBeenCalledTimes(1);
    expect(get).toHaveBeenCalledWith("/api/v2/home-summary");
  });
});
