import { useState } from "react";

interface ControlsProps {
  isConnected: boolean;
  eyeTrackerConnected: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
  onToggleMode: () => void;
  onClearFeedback: () => void;
  onTriggerFeedback: () => void;
  onConnectEyeTracker: () => void;
  onDisconnectEyeTracker: () => void;
  onSetCooldown: (cooldownSeconds: number) => void;
}

export function Controls({
  isConnected,
  eyeTrackerConnected,
  onConnect,
  onDisconnect,
  onToggleMode,
  onClearFeedback,
  onTriggerFeedback,
  onConnectEyeTracker,
  onDisconnectEyeTracker,
  onSetCooldown,
}: ControlsProps) {
  const [toggled, setToggled] = useState(false);

  return (
    <div className="section">
      <div className="section-title"
        style={{
          display: "flex",
          justifyContent: "space-between"
        }}
      >
        <span>Controls</span>
        <button className="btn small secondary toggle-btn" onClick={() => setToggled(!toggled)}>{toggled ? "Hide" : "Show"} </button>
      </div>
      {toggled && (
        <div className="controls">
          {!isConnected ? (
            <button className="btn" onClick={onConnect}>
              Connect Backend
            </button>
          ) : (
            <button className="btn secondary" onClick={onDisconnect}>
              Disconnect
            </button>
          )}

          {!eyeTrackerConnected ? (
            <button className="btn" onClick={onConnectEyeTracker} disabled={!isConnected}>
              Connect Eye Tracker
            </button>
          ) : (
            <button className="btn secondary" onClick={onDisconnectEyeTracker}>
              Disconnect Eye Tracker
            </button>
          )}

          <button className="btn secondary" onClick={onToggleMode} disabled={!isConnected}>
            Toggle Mode
          </button>

          <button className="btn secondary" onClick={onTriggerFeedback} disabled={!isConnected}>
            Trigger Feedback
          </button>

          <button className="btn secondary" onClick={onClearFeedback}>
            Clear Feedback
          </button>

          <div style={{ marginTop: "8px", borderTop: "1px solid var(--vscode-panel-border)", paddingTop: "8px" }}>
            <span style={{ fontSize: "12px", opacity: 0.8 }}>Cooldown:</span>
            <div style={{ display: "flex", gap: "4px", marginTop: "4px" }}>
              <button className="btn small secondary" onClick={() => onSetCooldown(300)} disabled={!isConnected}>
                5 min
              </button>
              <button className="btn small secondary" onClick={() => onSetCooldown(3600)} disabled={!isConnected}>
                Disable
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
