/**
 * Status bar manager - shows connection and system status.
 */
import * as vscode from "vscode";
import { SystemStatus } from "./types";

export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private currentStatus: SystemStatus = SystemStatus.DISCONNECTED;

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
    setStatus(status: SystemStatus): void {
        // TODO: Implement status update
    }

    /**
     * Update experiment status
     */
    setExperimentStatus(experimentStatus: string): void {
        // TODO: Implement experiment status update
    }

    /**
     * Set the connection state.
     */
    setConnected(connected: boolean): void {
        // TODO: Implement connection state update
    }

    /**
     * Set the operation mode.
     */
    setMode(mode: "reactive" | "proactive"): void {
        // TODO: Implement mode display
    }

    /**
     * Show temporary message.
     */
    showMessage(message: string, durationMs: number = 3000): void {
        // TODO: Implement temporary message
    }

    /**
     * Dispose of resources.
     */
    dispose(): void {
        this.statusBarItem.dispose();
    }

    // --- Private Methods ---

    private updateDisplay(): void {
        // TODO: Implement display update
        this.statusBarItem.text = "$(eye) Eye Tracking";
        this.statusBarItem.tooltip = "Click for status details";
    }

    private getStatusIcon(status: SystemStatus): string {
        // TODO: Implement icon mapping
        return "$(eye)";
    }

    private getStatusColor(status: SystemStatus): string | undefined {
        // TODO: Implement color mapping
        return undefined;
    }
}
