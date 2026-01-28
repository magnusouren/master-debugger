/**
 * Status bar manager - shows connection and system status.
 */
import * as vscode from "vscode";
import { SystemStatusMessage, SystemStatus } from "./types";

export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private currentStatus: SystemStatusMessage | null = null;
    private statusPanel: vscode.WebviewPanel | null = null;

    constructor(context: vscode.ExtensionContext) {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this.statusBarItem.command = "eyeTrackingDebugger.showStatus";
        context.subscriptions.push(this.statusBarItem);
        this.updateDisplay();
        this.statusBarItem.show();
    }

    /**
     * Update the displayed status.
     */
    setStatus(status: SystemStatusMessage): void {
        this.currentStatus = status;
        this.updateDisplay();

        if (this.statusPanel) {
            this.postStatusToPanel(status);
        }
    }

    /**
     * Set the connection state.
     */
    setConnected(connected: boolean): void {
        if (this.currentStatus) {
            this.currentStatus.vscode_connected = connected;
            this.updateDisplay();
            if (this.statusPanel) {
                this.postStatusToPanel(this.currentStatus);
            }            
        }
    }

    /**
     * Set the operation mode.
     */
    setMode(mode: "reactive" | "proactive"): void {
        if (this.currentStatus) {
            this.currentStatus.operation_mode = mode;
            this.updateDisplay();
            if (this.statusPanel) {
                this.postStatusToPanel(this.currentStatus);
            }            
        }
    }

    /**
     * Show temporary message.
     */
    showMessage(message: string, durationMs: number = 3000): void {
        const originalText = this.statusBarItem.text;
        this.statusBarItem.text = message;
        setTimeout(() => {
            this.statusBarItem.text = originalText;
        }, durationMs);
    }

    /**
     * Dispose of resources.
     */
    dispose(): void {
        this.statusBarItem.dispose();
    }

    async showStatusDetails(): Promise<void> {
        if (!this.currentStatus) {
            vscode.window.showInformationMessage("No status available");
            return;
        }

        // Reuse existing panel if it exists
        if (this.statusPanel) {
            this.statusPanel.reveal(this.statusPanel.viewColumn ?? vscode.ViewColumn.Active, true);
            this.postStatusToPanel(this.currentStatus);
            return;
        }

        // Create new panel
        this.statusPanel = vscode.window.createWebviewPanel(
            "eyeTrackingDebugger.status",
            "Eye Tracking Debugger – Status",
            { viewColumn: vscode.ViewColumn.Active, preserveFocus: true },
            {
                enableScripts: true, // needed for postMessage listener
                retainContextWhenHidden: true, // keeps state when switching tabs
            }
        );

        // Clean up on close
        this.statusPanel.onDidDispose(() => {
            this.statusPanel = null;
        });

        // Set initial HTML
        this.statusPanel.webview.html = this.getStatusPanelHtml(this.statusPanel.webview);

        // Send initial status
        this.postStatusToPanel(this.currentStatus);
    }

    // --- Private Methods ---

    private updateDisplay(): void {
        this.statusBarItem.backgroundColor = undefined;
        if (!this.currentStatus?.vscode_connected) {
            this.statusBarItem.text = "Eye Tracking Debugger: Disconnected";
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            this.statusBarItem.color = 'black';
            return;
        }

        const statusIcon = this.getStatusIcon(this.currentStatus.status);
        const statusColor = this.getStatusColor(this.currentStatus.status);
        const modeText =
            this.currentStatus.operation_mode === "proactive"
                ? "Proactive"
                : "Reactive";
        const connectionText = this.currentStatus.vscode_connected
            ? "Connected"
            : "Disconnected";
        const eyeTrackerText = this.currentStatus.eye_tracker_connected
            ? "Eye Tracker: Connected"
            : "Eye Tracker: Disconnected";
        
        const experimentText = this.currentStatus.experiment_active
            ? "Experiment: Running"
            : "Experiment: Stopped";

        this.statusBarItem.text = `Debugger: ${statusIcon} | ${connectionText} | ${modeText} | ${eyeTrackerText} | ${experimentText}`;
        this.statusBarItem.color = statusColor;
    }

    private getStatusIcon(status: SystemStatus): string {
        switch (status) {
            case SystemStatus.INITIALIZING:
                return "$(loading~spin)";
            case SystemStatus.READY:
                return "$(check)";
            case SystemStatus.PROCESSING:
                return "$(sync~spin)";
            case SystemStatus.DISCONNECTED:
                return "$(circle-slash)";
            case SystemStatus.PAUSED:
                return "$(debug-pause)";
            case SystemStatus.ERROR:
                return "$(error)";
            default:
                return "$(question)";
        }
    }

    private getStatusColor(status: SystemStatus): string {

        switch (status) {
             case SystemStatus.INITIALIZING:
                return "orange";
            case SystemStatus.READY:
                return "green";
            case SystemStatus.PROCESSING:
                return "yellow";
            case SystemStatus.DISCONNECTED:
                return "gray";
            case SystemStatus.PAUSED:
                return "blue";
            case SystemStatus.ERROR:
                return "red";
            default:
                return "gray";
        }
    }

    private postStatusToPanel(status: SystemStatusMessage): void {
        if (!this.statusPanel) return;

        // Timestamp fix (backend seconds -> JS ms)
        const tsMs = status.timestamp * 1000;

        this.statusPanel.webview.postMessage({
            type: "status_update",
            payload: {
                ...status,
                timestamp_ms: tsMs,
                time_local: new Date(tsMs).toLocaleString(),
            },
        });
    }

    private getStatusPanelHtml(webview: vscode.Webview): string {
        // AI-GENERATED: HTML content for simple status panel
        
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
        <title>Status</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                padding: 16px;
                line-height: 1.4;
            }
            .header {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 12px;
            }
            h2 { margin: 0; font-size: 18px; }
            .muted { opacity: 0.75; font-size: 12px; }
            .grid {
                display: grid;
                grid-template-columns: 180px 1fr;
                gap: 8px 12px;
                margin-top: 12px;
            }
            .label { opacity: 0.8; }
            .value { font-weight: 600; word-break: break-word; }
            .pill {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                border: 1px solid rgba(0,0,0,0.15);
            }
            .error {
                margin-top: 14px;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(255,0,0,0.35);
                background: rgba(255,0,0,0.08);
                white-space: pre-wrap;
                word-break: break-word;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: 12px;
            }
            .row { display: contents; }
            .divider {
                margin: 14px 0 10px;
                border-top: 1px solid rgba(0,0,0,0.12);
            }
        </style>
        </head>
        <body>
            <div class="header">
                <h2>Eye Tracking Debugger – Status</h2>
                <div class="muted" id="last_updated">Last updated: –</div>
            </div>

            <div class="grid" id="grid">
                <div class="label">Connection</div>
                <div class="value"><span class="pill" id="connection_pill">–</span></div>
                <div class="label">Status</div>
                <div class="value"><span class="pill" id="status_pill">–</span></div>

                <div class="label">Time</div>
                <div class="value" id="time_local">–</div>

                <div class="label">Mode</div>
                <div class="value" id="mode">–</div>

                <div class="label">VS Code</div>
                <div class="value" id="vscode_connected">–</div>

                <div class="label">Eye Tracker</div>
                <div class="value" id="eye_tracker_connected">–</div>

                <div class="label">Experiment</div>
                <div class="value" id="experiment_active">–</div>

                <div class="label">LLM Model</div>
                <div class="value" id="llm_model">–</div>

                <div class="label">Experiment ID</div>
                <div class="value" id="experiment_id">–</div>

                <div class="label">Participant ID</div>
                <div class="value" id="participant_id">–</div>

                <div class="label">Samples processed</div>
                <div class="value" id="samples_processed">–</div>

                <div class="label">Feedback generated</div>
                <div class="value" id="feedback_generated">–</div>
            </div>

            <div class="divider"></div>

            <div id="error_box" class="error" style="display:none;"></div>

        <script>
            const statusPill = document.getElementById("status_pill");
            const connectionPill = document.getElementById("connection_pill");
            const lastUpdated = document.getElementById("last_updated");

            function setText(id, value) {
                const el = document.getElementById(id);
                if (!el) return;
                el.textContent = (value === null || value === undefined || value === "") ? "–" : String(value);
            }

            function yesNo(x) {
                return x ? "Connected ✅" : "Disconnected ❌";
            }

            function runningStopped(x) {
                return x ? "Running ✅" : "NOT Running ❌";
            }

            function pillForStatus(status) {
                const s = String(status || "unknown").toUpperCase();
                statusPill.textContent = s;

                // light, neutral styling; keep it simple
                // (you can map colors if you want)
                statusPill.style.borderColor = "rgba(0,0,0,0.20)";
                statusPill.style.background = "rgba(0,0,0,0.06)";

                if (s === "ERROR") {
                    statusPill.style.borderColor = "rgba(255,0,0,0.40)";
                    statusPill.style.background = "rgba(255,0,0,0.10)";
                } else if (s === "READY") {
                    statusPill.style.borderColor = "rgba(0,128,0,0.35)";
                    statusPill.style.background = "rgba(0,128,0,0.10)";
                } else if (s === "PROCESSING") {
                    statusPill.style.borderColor = "rgba(200,160,0,0.45)";
                    statusPill.style.background = "rgba(200,160,0,0.12)";
                }
            }

            function pillForConnection(connected) {
                if (connected) {
                    connectionPill.textContent = "CONNECTED";
                    connectionPill.style.borderColor = "rgba(0,128,0,0.35)";
                    connectionPill.style.background = "rgba(0,128,0,0.10)";
                } else {
                    connectionPill.textContent = "DISCONNECTED";
                    connectionPill.style.borderColor = "rgba(255,0,0,0.40)";
                    connectionPill.style.background = "rgba(255,0,0,0.10)";
                }
            }

            window.addEventListener("message", (event) => {
                const msg = event.data;
                if (!msg || msg.type !== "status_update") return;

                pillForConnection(!!msg.payload.vscode_connected);

                const s = msg.payload;

                if (!s.vscode_connected) {
                    // If disconnected, clear most fields
                    setText("time_local", "–");
                    setText("mode", "–");
                    setText("vscode_connected", "–");
                    setText("eye_tracker_connected", "–");
                    setText("experiment_active", "–");
                    setText("experiment_id", "–");
                    setText("participant_id", "–");
                    setText("samples_processed", "–");
                    setText("feedback_generated", "–");

                    statusPill.textContent = "DISCONNECTED";
                    statusPill.style.borderColor = "rgba(255,0,0,0.40)";
                    statusPill.style.background = "rgba(255,0,0,0.10)";

                    const errorBox = document.getElementById("error_box");
                    errorBox.style.display = "none";
                    errorBox.textContent = "";

                    lastUpdated.textContent = "Last updated: " + new Date().toLocaleTimeString();
                    return;
                }

                pillForStatus(s.status);
                setText("time_local", s.time_local);
                setText("mode", s.operation_mode);
                setText("vscode_connected", yesNo(!!s.vscode_connected));
                setText("eye_tracker_connected", yesNo(!!s.eye_tracker_connected));
                setText("experiment_active", runningStopped(!!s.experiment_active));
                setText("llm_model", s.llm_model);
                setText("experiment_id", s.experiment_id);
                setText("participant_id", s.participant_id);
                setText("samples_processed", s.samples_processed);
                setText("feedback_generated", s.feedback_generated);

                const errorBox = document.getElementById("error_box");
                if (s.error_message) {
                    errorBox.style.display = "block";
                    errorBox.textContent = s.error_message;
                } else {
                    errorBox.style.display = "none";
                    errorBox.textContent = "";
                }

                lastUpdated.textContent = "Last updated: " + new Date().toLocaleTimeString();
            });
        </script>
        </body>
        </html>`;
    }
}
