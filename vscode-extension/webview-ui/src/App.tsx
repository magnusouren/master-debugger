import { useState, useEffect, useCallback } from "react";
import { FeedbackList } from "./components/FeedbackList";
import { StatusPanel } from "./components/StatusPanel";
import { Controls } from "./components/Controls";
import { vscode } from "./utilities/vscode";
import type { FeedbackItem, SystemStatus } from "./types";

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
    vscode.postMessage({ type: "toggleMode" });
  };

  const handleClearFeedback = () => {
    vscode.postMessage({ type: "clearFeedback" });
  };

  const handleTriggerFeedback = () => {
    vscode.postMessage({ type: "triggerFeedback" });
  };

  const handleFeedbackInteraction = (feedbackId: string, interactionType: "dismissed" | "accepted") => {
    setFeedbackItems((prevItems) => 
      prevItems.filter(item => item.metadata.feedback_id !== feedbackId)
    );
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
      <header className="header">
        <h1>Eye Tracking Debugger</h1>
        <span className={`status-badge ${isConnected ? "connected" : "disconnected"}`}>
          <span className="status-dot" />
          {isConnected ? "Connected" : "Disconnected"}
        </span>
      </header>

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

      {status && <StatusPanel status={status} />}

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
