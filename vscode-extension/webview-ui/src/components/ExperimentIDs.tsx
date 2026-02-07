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
  const [errorMessage, setErrorMessage] = useState("");

  const handleStartExperiment = async () => {
    if (!experimentId || !participantId) {
      setErrorMessage("Experiment ID and Participant ID are required.");
      return;
    }
    try {
      await startExperiment(experimentId, participantId);
    } catch (error) {
      setErrorMessage("Failed to start experiment. Please check connection and try again.");
    }
  }


  return (
    <div className="section">
      <div className="section-title"
        style={{
          display: "flex",
          justifyContent: "space-between"
        }}
      >
        <span>Experiment Control</span>
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
            <label htmlFor="participant-id-input">Participant ID:</label>
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
          {errorMessage && <p className="error-message">{errorMessage}</p>}
        </div>
      )}
    </div>
  );
}