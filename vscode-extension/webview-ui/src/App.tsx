import { useState, useEffect, useCallback } from "react";
import { FeedbackList } from "./components/FeedbackList";
import { StatusPanel } from "./components/StatusPanel";
import { Controls } from "./components/Controls";
import { vscode } from "./utilities/vscode";
import type { FeedbackItem, SystemStatus, InteractionType } from "./types";
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

export function App() {
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

  const handleFeedbackInteraction = (feedbackId: string, interactionType: InteractionType) => {
    // Remove from list when rejected or dismissed
    if (interactionType === "rejected" || interactionType === "dismissed") {
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

  const handleSetCooldown = (cooldownSeconds: number) => {
    vscode.postMessage({
      type: "setCooldown",
      payload: { cooldownSeconds }
    });
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
        <div className="cooldown-buttons">


          {status?.feedback_cooldown_left_s && status?.feedback_cooldown_left_s > 80000 ? (
            <button className="btn small secondary" onClick={() => handleSetCooldown(15)} disabled={!isConnected}>
              Enable Feedback
            </button>
          ) : (
            <>
              <button className="btn small secondary" onClick={() => handleSetCooldown(300)} disabled={!isConnected}>
                Disable for 5 min
              </button>
              <button className="btn small secondary" onClick={() => handleSetCooldown(86400)} disabled={!isConnected}>
                Disable Feedback
              </button>
            </>
          )}
        </div>

        <FeedbackList
          items={feedbackItems}
          onInteraction={handleFeedbackInteraction}
        />
      </div>


    </div>
  );
}

