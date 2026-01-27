/**
 * Status bar manager - shows connection and system status.
 */
import * as vscode from "vscode";
import { SystemStatusMessage, SystemStatus } from "./types";

export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private currentStatus: SystemStatusMessage | null = null;

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
    }

    /**
     * Set the connection state.
     */
    setConnected(connected: boolean): void {
        if (this.currentStatus) {
            this.currentStatus.vscode_connected = connected;
            this.updateDisplay();
        }
    }

    /**
     * Set the operation mode.
     */
    setMode(mode: "reactive" | "proactive"): void {
        if (this.currentStatus) {
            this.currentStatus.operation_mode = mode;
            this.updateDisplay();
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
}
