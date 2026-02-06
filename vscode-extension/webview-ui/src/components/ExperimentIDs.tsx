import { useState } from "react";

interface ExperimentIDsProps {
  experimentIsRunning: boolean;
  startExperiment: (experimentId: string, participantId: string) => Promise<void>;
  endExperiment: () => void;
}

export function ExperimentIDs({
  experimentIsRunning,
  startExperiment,
  endExperiment

}: ExperimentIDsProps) {
  const [toggled, setToggled] = useState(false);
  const [experimentId, setExperimentId] = useState("");
  const [participantId, setParticipantId] = useState("");

  const handleStartExperiment = () => {
    if (!experimentId || !participantId) {
      alert("Please enter both Experiment ID and Participant ID to start the experiment.");
      return;
    }
    startExperiment(experimentId, participantId);
  }


  return (
    <div className="section">
      <div className="section-title"
        style={{
          display: "flex",
          justifyContent: "space-between"
        }}
      >
        <span>Experiment Controll</span>
        <button className="btn small secondary toggle-btn" onClick={() => setToggled(!toggled)}>{toggled ? "Hide" : "Show"}</button>
      </div>
      {toggled && (
        <div className="experiment-settings">
          <div className="experiment-setting">
            <label htmlFor="experiment-id-input">Experiment ID:</label>
            <input
              type="text"
              id="experiment-id-input"
              placeholder="Enter experiment ID"
              className="text-input"
              value={experimentId}
              onChange={(e) => setExperimentId(e.target.value)}
              disabled={experimentIsRunning}
            />
          </div>
          <div className="experiment-setting">
            <label htmlFor="experiment-description-input">Participant ID:</label>
            <input
              type="text"
              id="participant-id-input"
              placeholder="Enter participant ID"
              className="text-input"
              value={participantId}
              onChange={(e) => setParticipantId(e.target.value)}
              disabled={experimentIsRunning}
            />
          </div>
          <div className="experiment-setting">
            {!experimentIsRunning ? (
              <button className="btn" onClick={handleStartExperiment}>
                Start Experiment
              </button>
            ) : (
              <button className="btn secondary" onClick={endExperiment}>
                End Experiment
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}