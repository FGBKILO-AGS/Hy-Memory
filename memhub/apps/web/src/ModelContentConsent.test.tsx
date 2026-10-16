// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as api from "./api";
import { ModelContentConsent } from "./ModelContentConsent";

vi.mock("./api", () => ({
  getModelContentPermission: vi.fn(),
  setModelContentPermission: vi.fn(),
}));
vi.mock("./ui-trace", () => ({
  traceUiAction: () => ({ traceId: "trace-id", done: vi.fn() }),
}));

beforeEach(() => {
  vi.mocked(api.getModelContentPermission).mockResolvedValue({
    allowed: false,
  });
  vi.mocked(api.setModelContentPermission).mockImplementation(
    async (allowed) => ({
      allowed,
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("shows one global model-content switch", async () => {
  render(<ModelContentConsent onClose={vi.fn()} />);

  const toggle = await screen.findByRole("switch", {
    name: "允许当前配置模型读取本地会话内容",
  });
  expect(toggle.getAttribute("aria-checked")).toBe("false");
  expect(screen.queryByText(/选择来源|授权范围|后续新增/)).toBeNull();
});

it("writes the desired boolean and updates the switch", async () => {
  const user = userEvent.setup();
  render(<ModelContentConsent onClose={vi.fn()} />);
  const toggle = await screen.findByRole("switch", {
    name: "允许当前配置模型读取本地会话内容",
  });

  await user.click(toggle);

  await waitFor(() =>
    expect(api.setModelContentPermission).toHaveBeenCalledWith(
      true,
      "trace-id",
    ),
  );
  expect(toggle.getAttribute("aria-checked")).toBe("true");
});

it("states that generated-memory search is independent", async () => {
  render(<ModelContentConsent onClose={vi.fn()} />);
  expect(
    await screen.findByText(/已生成记忆的搜索和读取不受影响/),
  ).toBeTruthy();
});
