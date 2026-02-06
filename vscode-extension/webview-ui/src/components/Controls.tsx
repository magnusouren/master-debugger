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
        </div>
      )}
    </div>
  );
}
