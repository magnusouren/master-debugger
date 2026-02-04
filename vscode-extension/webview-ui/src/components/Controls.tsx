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
  return (
    <div className="section">
      <div className="section-title">Controls</div>
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
    </div>
  );
}
