/**
 * Status bar manager - shows connection and system status.
 */
import * as vscode from "vscode";
import { SystemStatusMessage, SystemStatus } from "./types";

export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private connectedToBackend: boolean = false;
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
        this.connectedToBackend = connected;
        this.updateDisplay();

        if (this.statusPanel && this.currentStatus) {
            this.postStatusToPanel(this.currentStatus);
        }
    }

    /**
     * Set the operation mode.
     */
    setMode(mode: "reactive" | "proactive"): void {
        if (!this.currentStatus) return;

        this.currentStatus.operation_mode = mode;
        this.updateDisplay();

        if (this.statusPanel) {
            this.postStatusToPanel(this.currentStatus);
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

        if (this.statusPanel) {
            this.statusPanel.reveal(this.statusPanel.viewColumn ?? vscode.ViewColumn.Active, true);
            this.postStatusToPanel(this.currentStatus);
            return;
        }

        this.statusPanel = vscode.window.createWebviewPanel(
            "eyeTrackingDebugger.status",
            "Eye Tracking Debugger – Status",
            { viewColumn: vscode.ViewColumn.Active, preserveFocus: true },
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            }
        );

        this.statusPanel.onDidDispose(() => {
            this.statusPanel = null;
        });

        this.statusPanel.webview.html = this.getStatusPanelHtml(this.statusPanel.webview);
        this.postStatusToPanel(this.currentStatus);
    }

    // ---------------------------------------------------------------------
    // Status bar
    // ---------------------------------------------------------------------

    private updateDisplay(): void {
        this.statusBarItem.backgroundColor = undefined;
        this.statusBarItem.color = undefined;

        if (!this.connectedToBackend || !this.currentStatus) {
            this.statusBarItem.text = "Eye Tracking Debugger: Disconnected";
            this.statusBarItem.backgroundColor =
                new vscode.ThemeColor("statusBarItem.errorBackground");
            this.statusBarItem.color = "black";
            return;
        }

        const statusIcon = this.getStatusIcon(this.currentStatus.status);
        const modeText =
            this.currentStatus.operation_mode === "proactive"
                ? "Proactive"
                : "Reactive";

        const eyeTrackerText = this.currentStatus.eye_tracker_connected
            ? "Eye Tracker: Connected"
            : "Eye Tracker: Disconnected";

        const experimentText = this.currentStatus.experiment_active
            ? "Experiment: Running"
            : "Experiment: Stopped";

        this.statusBarItem.text =
            `Debugger: ${statusIcon} | Connected | ${modeText} | ${eyeTrackerText} | ${experimentText}`;

        this.statusBarItem.color = this.getStatusColor(this.currentStatus.status);
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
            case SystemStatus.PAUSED:
                return "blue";
            case SystemStatus.ERROR:
                return "red";
            default:
                return "gray";
        }
    }

    // ---------------------------------------------------------------------
    // Webview
    // ---------------------------------------------------------------------

    private postStatusToPanel(status: SystemStatusMessage): void {
        if (!this.statusPanel) return;

        const tsMs = status.timestamp * 1000;

        this.statusPanel.webview.postMessage({
            type: "status_update",
            payload: {
                ...status,
                connected_to_backend: this.connectedToBackend,
                timestamp_ms: tsMs,
                time_local: new Date(tsMs).toLocaleString(),
            },
        });
    }

    private getStatusPanelHtml(_: vscode.Webview): string {
        return `<!DOCTYPE html>
                <html lang="en">
                <head>
                <meta charset="UTF-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <meta http-equiv="Content-Security-Policy"
                    content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
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
                    margin-bottom: 12px;
                }
                h2 { margin: 0; font-size: 18px; }
                .muted { opacity: 0.75; font-size: 12px; }
                .grid {
                    display: grid;
                    grid-template-columns: 180px 1fr;
                    gap: 8px 12px;
                }
                .label { opacity: 0.8; }
                .value { font-weight: 600; }
                .pill {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 999px;
                    font-size: 12px;
                    font-weight: 700;
                    border: 1px solid rgba(0,0,0,0.2);
                }
                .divider {
                    margin: 14px 0 10px;
                    border-top: 1px solid rgba(0,0,0,0.12);
                }
                .error {
                    margin-top: 14px;
                    padding: 10px;
                    border-radius: 8px;
                    border: 1px solid rgba(255,0,0,0.35);
                    background: rgba(255,0,0,0.08);
                    font-family: ui-monospace, monospace;
                    font-size: 12px;
                    white-space: pre-wrap;
                }
                </style>
                </head>
                <body>
                <div class="header">
                    <h2>Eye Tracking Debugger – Status</h2>
                    <div class="muted" id="last_updated">Last updated: –</div>
                </div>

                <div class="grid">
                    <div class="label">Connection</div>
                    <div class="value"><span class="pill" id="connection_pill">–</span></div>

                    <div class="label">Status</div>
                    <div class="value"><span class="pill" id="status_pill">–</span></div>

                    <div class="label">Time</div>
                    <div class="value" id="time_local">–</div>

                    <div class="label">Mode</div>
                    <div class="value" id="mode">–</div>

                    <div class="label">Eye Tracker</div>
                    <div class="value" id="eye_tracker_connected">–</div>

                    <div class="label">Experiment</div>
                    <div class="value" id="experiment_active">–</div>

                    <div class="label">Experiment ID</div>
                    <div class="value" id="experiment_id">–</div>

                    <div class="label">Participant ID</div>
                    <div class="value" id="participant_id">–</div>

                    <div class="label">Eye samples processed</div>
                    <div class="value" id="eye_samples_processed">–</div>

                    <div class="label">Code window samples processed</div>
                    <div class="value" id="code_window_samples_processed">–</div>

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
                    el.textContent = value ?? "–";
                }

                function pillForConnection(connected) {
                    connectionPill.textContent = connected ? "CONNECTED" : "DISCONNECTED";
                    connectionPill.style.borderColor = connected
                        ? "rgba(0,128,0,0.35)"
                        : "rgba(255,0,0,0.40)";
                    connectionPill.style.background = connected
                        ? "rgba(0,128,0,0.10)"
                        : "rgba(255,0,0,0.10)";
                }

                function pillForStatus(status) {
                    const s = String(status || "UNKNOWN").toUpperCase();
                    statusPill.textContent = s;
                }

                window.addEventListener("message", (event) => {
                    const msg = event.data;
                    if (!msg || msg.type !== "status_update") return;

                    const s = msg.payload;
                    const connected = !!s.connected_to_backend;

                    pillForConnection(connected);
                    lastUpdated.textContent = "Last updated: " + new Date().toLocaleTimeString();

                    if (!connected) {
                        pillForStatus("DISCONNECTED");
                        return;
                    }

                    pillForStatus(s.status);
                    setText("time_local", s.time_local);
                    setText("mode", s.operation_mode);
                    setText("eye_tracker_connected", s.eye_tracker_connected ? "Connected" : "Disconnected");
                    setText("experiment_active", s.experiment_active ? "Running" : "Stopped");
                    setText("experiment_id", s.experiment_id);
                    setText("participant_id", s.participant_id);
                    setText("eye_samples_processed", s.eye_samples_processed);
                    setText("code_window_samples_processed", s.code_window_samples_processed);
                    setText("feedback_generated", s.feedback_generated);

                    const errorBox = document.getElementById("error_box");
                    if (s.error_message) {
                        errorBox.style.display = "block";
                        errorBox.textContent = s.error_message;
                    } else {
                        errorBox.style.display = "none";
                        errorBox.textContent = "";
                    }
                });
                </script>
                </body>
                </html>`;
    }
}