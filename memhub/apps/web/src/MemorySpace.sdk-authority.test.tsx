// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MemorySpace } from "./MemorySpace";

const api = vi.hoisted(() => ({
  authority: vi.fn(),
  legacyLibrary: vi.fn(),
  graph: vi.fn(),
}));
vi.mock("./api", () => ({
  getMemoryAuthority: api.authority,
  getMemoryLibrary: api.legacyLibrary,
  getMemories: async () => [],
  getMemoryGraph: api.graph,
  getMemoryProcessing: async () => null,
  getModelProfiles: async () => [],
  apiErrorCode: () => null,
}));
vi.mock("./GraphWidthPrompt", () => ({ GraphWidthPrompt: () => null }));
vi.mock("./MemoryNoteComposer", () => ({ MemoryNoteComposer: () => null }));
vi.mock("./MemoryMap", () => ({
  MemoryMap: ({ hidden }: { hidden: boolean }) => (
    <div hidden={hidden} data-testid="legacy-map" />
  ),
}));
vi.mock("./SdkMemoryLibrary", () => ({
  SdkMemoryLibrary: () => <div data-testid="sdk-library" />,
}));
vi.mock("./MemoryRecallScope", () => ({
  MemoryRecallScope: () => null,
  memoryProvenanceLabel: () => "原话",
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("opens the SDK library on SDK authority instead of a map that cannot load", async () => {
  api.authority.mockResolvedValue({ authority: "sdk", migration_id: "m" });

  render(<MemorySpace simplified active />);

  const library = await screen.findByTestId("sdk-library");
  expect(library.closest("[hidden]")).toBeNull();
  expect(screen.getByTestId("legacy-map").hidden).toBe(true);
  expect(api.legacyLibrary).not.toHaveBeenCalled();
  expect(api.graph).not.toHaveBeenCalled();
  expect(screen.queryByText("正在读取…")).toBeNull();
});

it("keeps asking for the authority until Core answers", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    api.authority
      .mockRejectedValueOnce(new Error("core_not_ready"))
      .mockRejectedValueOnce(new Error("core_not_ready"))
      .mockResolvedValue({ authority: "sdk", migration_id: "m" });

    render(<MemorySpace simplified active />);
    await vi.advanceTimersByTimeAsync(2_000);

    await waitFor(() => expect(api.authority).toHaveBeenCalledTimes(3));
    expect(await screen.findByTestId("sdk-library")).toBeTruthy();
  } finally {
    vi.useRealTimers();
  }
});
