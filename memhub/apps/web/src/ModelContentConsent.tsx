import { RefreshCw, X } from "lucide-react";
import { useEffect, useId, useRef } from "react";
import { R6Button, R6Switch } from "./R6Controls";
import { useModelContentPermission } from "./use-model-content-permission";
import "./model-content-consent.css";

export type ModelContentConsentProps = {
  onClose(): void;
  onConfigureModel?(): void;
};

export function ModelContentConsent({
  onClose,
  onConfigureModel,
}: ModelContentConsentProps) {
  const id = useId();
  const panel = useRef<HTMLElement>(null);
  const permission = useModelContentPermission();

  useEffect(() => {
    const before = document.activeElement;
    panel.current?.querySelector<HTMLElement>("button:not(:disabled)")?.focus();
    return () => {
      if (before instanceof HTMLElement && before.isConnected) before.focus();
    };
  }, []);

  return (
    <div className="consent-backdrop model-content-backdrop">
      <section
        ref={panel}
        className="consent-sheet model-content-consent"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${id}-title`}
        aria-busy={permission.reading || permission.writing}
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape" && !permission.writing) onClose();
        }}
      >
        <header className="model-content-heading">
          <div>
            <p className="eyebrow">模型内容权限</p>
            <h2 id={`${id}-title`}>允许模型读取本地会话内容</h2>
          </div>
          <R6Button
            variant="quiet"
            aria-label="刷新模型内容权限"
            title="刷新模型内容权限"
            disabled={permission.reading || permission.writing}
            onClick={() => void permission.refresh()}
          >
            <RefreshCw size={18} aria-hidden="true" />
          </R6Button>
          <R6Button
            variant="quiet"
            aria-label="关闭"
            title="关闭"
            disabled={permission.writing}
            onClick={onClose}
          >
            <X size={18} aria-hidden="true" />
          </R6Button>
        </header>
        <div className="model-content-body">
          <p>
            开启后，当前配置的模型可以读取已导入和之后同步到本机的会话内容，用于生成记忆。
          </p>
          <p className="model-content-meta">
            来源同步决定哪些会话进入本机；此开关只控制模型是否可以读取这些内容。
            已生成记忆的搜索和读取不受影响。
          </p>
          <div className="model-content-choice model-content-future">
            <span>允许当前配置模型读取本地会话内容</span>
            <R6Switch
              label="允许当前配置模型读取本地会话内容"
              checked={permission.allowed === true}
              disabled={
                permission.allowed === null ||
                permission.reading ||
                permission.writing
              }
              onCheckedChange={(allowed) => void permission.update(allowed)}
            />
          </div>
          {permission.reading && <p role="status">正在读取当前设置…</p>}
          {permission.writing && <p role="status">正在保存设置…</p>}
          {permission.error && (
            <p className="model-content-error" role="alert">
              {permission.error}
            </p>
          )}
          {permission.allowed === false && onConfigureModel && (
            <p className="model-content-meta">
              可先配置并测试模型，再决定是否允许读取会话内容。
            </p>
          )}
        </div>
        <footer className="consent-actions model-content-actions">
          {onConfigureModel && (
            <R6Button
              variant="secondary"
              disabled={permission.writing}
              onClick={onConfigureModel}
            >
              配置模型
            </R6Button>
          )}
          <R6Button
            variant="primary"
            disabled={permission.writing}
            onClick={onClose}
          >
            完成
          </R6Button>
        </footer>
      </section>
    </div>
  );
}
