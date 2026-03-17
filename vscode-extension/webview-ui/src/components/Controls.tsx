import { useState } from "react";
import { OperationMode } from "../types";

interface ControlsProps {
  isConnected: boolean;
  eyeTrackerConnected: boolean;
  currentMode: OperationMode | null;
  onConnect: () => void;
  onDisconnect: () => void;
  onSetMode: (mode: OperationMode) => void;
  onClearFeedback: () => void;
  onTriggerFeedback: () => void;
  onConnectEyeTracker: () => void;
  onDisconnectEyeTracker: () => void;
}

export function Controls({
  isConnected,
  eyeTrackerConnected,
  currentMode,
  onConnect,
  onDisconnect,
  onSetMode,
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


          <button className="btn secondary" onClick={onTriggerFeedback} disabled={!isConnected}>
            Trigger Feedback
          </button>

          <div className="mode-buttons">
            {[OperationMode.CONTROL, OperationMode.PROACTIVE, OperationMode.REACTIVE, OperationMode.QUESTIONNAIRE].map((mode) => {
              const isActive = currentMode === mode;
              const label = mode.charAt(0).toUpperCase() + mode.slice(1);
              return (
                <button
                  key={mode}
                  className={`btn ${isActive ? "" : "secondary"}`}
                  onClick={() => onSetMode(mode)}
                  disabled={!isConnected || isActive}
                  aria-pressed={isActive}
                >
                  {label}
                </button>
              );
            })}
          </div>


          <button className="btn secondary" onClick={onClearFeedback}>
            Clear Feedback
          </button>
        </div>
      )}
    </div>
  );
}
