import type { SystemStatus } from "../types";

interface StatusPanelProps {
  status: SystemStatus;
}

export function StatusPanel({ status }: StatusPanelProps) {
  return (
    <div className="section">
      <div className="section-title">System Status</div>
      <div className="status-info">
        <span className="label">Status</span>
        <span className="value">{status.status}</span>
        
        <span className="label">Mode</span>
        <span className="value">{status.operation_mode}</span>
        
        <span className="label">Eye Tracker</span>
        <span className="value">{status.eye_tracker_connected ? "✓ Connected" : "✗ Disconnected"}</span>
        
        <span className="label">Eye Samples</span>
        <span className="value">{status.eye_samples_processed}</span>
        
        <span className="label">Code Samples</span>
        <span className="value">{status.code_window_samples_processed}</span>
        
        <span className="label">Feedback Generated</span>
        <span className="value">{status.feedback_generated}</span>
        
        {status.experiment_active && (
          <>
            <span className="label">Experiment</span>
            <span className="value">{status.experiment_id || "Active"}</span>
          </>
        )}
        
        {status.feedback_cooldown_left_s > 0 && (
          <>
            <span className="label">Cooldown</span>
            <span className="value">{status.feedback_cooldown_left_s.toFixed(1)}s</span>
          </>
        )}
        
        {status.llm_model && (
          <>
            <span className="label">LLM Model</span>
            <span className="value">{status.llm_model}</span>
          </>
        )}
      </div>
    </div>
  );
}
