import { useState, useEffect, useCallback } from "react";
import { FeedbackList } from "./components/FeedbackList";
import { StatusPanel } from "./components/StatusPanel";
import { Controls } from "./components/Controls";
import { vscode } from "./utilities/vscode";
import type { FeedbackItem, SystemStatus } from "./types";
import { ExperimentIDs } from "./components/ExperimentIDs";

// Message types from extension to webview
interface ConnectionStatusMessage {
  type: "connectionStatus";
  payload: { connected: boolean };
}

interface StatusUpdateMessage {
  type: "statusUpdate";
  payload: SystemStatus;
}

interface FeedbackUpdateMessage {
  type: "feedbackUpdate";
  payload: { items: FeedbackItem[] };
}

interface ClearFeedbackMessage {
  type: "clearFeedback";
  payload: Record<string, never>;
}

type ExtensionMessage = ConnectionStatusMessage | StatusUpdateMessage | FeedbackUpdateMessage | ClearFeedbackMessage;

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [feedbackItems, setFeedbackItems] = useState<FeedbackItem[]>([]);

  // Handle messages from the extension
  const handleMessage = useCallback((event: MessageEvent<ExtensionMessage>) => {
    const message = event.data;

    switch (message.type) {
      case "connectionStatus":
        setIsConnected(message.payload.connected);
        break;
      case "statusUpdate":
        setStatus(message.payload);
        break;
      case "feedbackUpdate":
        setFeedbackItems(message.payload.items);
        break;
      case "clearFeedback":
        setFeedbackItems([]);
        break;
    }
  }, []);

  useEffect(() => {
    window.addEventListener("message", handleMessage);

    // Request initial state from extension
    vscode.postMessage({ type: "ready" });

    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, [handleMessage]);

  const handleConnect = () => {
    vscode.postMessage({ type: "connect" });
  };

  const handleDisconnect = () => {
    vscode.postMessage({ type: "disconnect" });
  };

  const handleToggleMode = () => {
    if (!status) return;
    if (status.operation_mode === "reactive") {
      vscode.postMessage({
        type: "toggleMode",
        payload: { new_mode: "proactive" }
      });
      return;
    }
    vscode.postMessage({
      type: "toggleMode",
      payload: { new_mode: "reactive" }
    });
  };

  const handleClearFeedback = () => {
    vscode.postMessage({ type: "clearFeedback" });
  };

  const handleTriggerFeedback = () => {
    vscode.postMessage({ type: "triggerFeedback" });
  };

  const handleFeedbackInteraction = (feedbackId: string, interactionType: "dismissed" | "accepted") => {
    if (interactionType === "dismissed") {
      setFeedbackItems((prevItems) =>
        prevItems.filter(item => item.metadata.feedback_id !== feedbackId)
      );
    }
    vscode.postMessage({
      type: "feedbackInteraction",
      payload: { feedbackId, interactionType }
    });
  };

  const handleConnectEyeTracker = () => {
    vscode.postMessage({ type: "connectEyeTracker" });
  };

  const handleDisconnectEyeTracker = () => {
    vscode.postMessage({ type: "disconnectEyeTracker" });
  };

  return (
    <div className="app">
      <div className="controller-section">
        <header className="header">
          <h1>Eye Tracking Debugger</h1>
          <span className={`status-badge ${isConnected ? "connected" : "disconnected"}`}>
            <span className="status-dot" />
            {isConnected ? "Connected" : "Disconnected"}
          </span>
        </header>

        <div className="section">
          <ExperimentIDs
            experimentIsRunning={status?.experiment_active ?? false}
            startExperiment={async (experimentId: string, participantId: string) => {
              vscode.postMessage({
                type: "startExperiment",
                payload: { experimentId, participantId }
              });
            }}
            endExperiment={() => {
              vscode.postMessage({ type: "endExperiment" });
            }}
          />
        </div>

        <div className="section">
          <Controls
            isConnected={isConnected}
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
            onToggleMode={handleToggleMode}
            onClearFeedback={handleClearFeedback}
            onTriggerFeedback={handleTriggerFeedback}
            onConnectEyeTracker={handleConnectEyeTracker}
            onDisconnectEyeTracker={handleDisconnectEyeTracker}
            eyeTrackerConnected={status?.eye_tracker_connected ?? false}
          />
        </div>
        {status &&
          <div className="section">
            <StatusPanel status={status} />
          </div>
        }
      </div>

      <div className="section">
        <div className="section-title">Feedback</div>
        <FeedbackList
          items={feedbackItems}
          onInteraction={handleFeedbackInteraction}
        />
      </div>
    </div>
  );
}

export default App;
