import { useCallback, useEffect, useRef, useState } from "react";
import { getModelContentPermission, setModelContentPermission } from "./api";
import { traceUiAction } from "./ui-trace";
import { isDefiniteRefusal, writeTraceStatus } from "./write-outcome";

export function useModelContentPermission(enabled = true) {
  const [allowed, setAllowed] = useState<boolean | null>(null);
  const [reading, setReading] = useState(enabled);
  const [writing, setWriting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(false);
  const version = useRef(0);
  const flight = useRef(false);

  const refresh = useCallback(async () => {
    if (!enabled) return;
    const readVersion = ++version.current;
    setReading(true);
    try {
      const permission = await getModelContentPermission();
      if (!alive.current || readVersion !== version.current) return;
      setAllowed(permission.allowed);
      setError(null);
    } catch {
      if (!alive.current || readVersion !== version.current) return;
      setError("暂时无法读取模型内容权限，请重试。");
    } finally {
      if (alive.current && readVersion === version.current) setReading(false);
    }
  }, [enabled]);

  useEffect(() => {
    alive.current = true;
    if (enabled) void refresh();
    return () => {
      alive.current = false;
      version.current += 1;
    };
  }, [enabled, refresh]);

  const update = useCallback(
    async (next: boolean) => {
      if (!enabled || flight.current || allowed === null || next === allowed)
        return false;
      flight.current = true;
      version.current += 1;
      setWriting(true);
      setError(null);
      const trace = traceUiAction("settings", "toggle");
      try {
        const permission = await setModelContentPermission(next, trace.traceId);
        trace.done("succeeded");
        if (alive.current) setAllowed(permission.allowed);
        return true;
      } catch (reason) {
        trace.done(writeTraceStatus(reason), {
          reason: isDefiniteRefusal(reason)
            ? "model_content_permission_refused"
            : "model_content_permission_unknown",
        });
        if (alive.current) {
          setError(
            isDefiniteRefusal(reason)
              ? "模型内容权限未能修改，请重试。"
              : "修改结果尚未确认，正在重新读取当前设置。",
          );
          await refresh();
        }
        return false;
      } finally {
        flight.current = false;
        if (alive.current) setWriting(false);
      }
    },
    [allowed, enabled, refresh],
  );

  return { allowed, reading, writing, error, refresh, update };
}
