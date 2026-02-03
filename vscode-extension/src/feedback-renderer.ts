/**
 * Feedback renderer - displays feedback in the VS Code editor.
 */
import * as vscode from "vscode";
import {
    FeedbackItem,
    FeedbackType,
    FeedbackPriority,
    CodeRange,
    FeedbackInteraction,
} from "./types";

export type InteractionCallback = (interaction: FeedbackInteraction) => void;

export class FeedbackRenderer {
    private decorationType: vscode.TextEditorDecorationType | null = null;
    private diagnosticCollection: vscode.DiagnosticCollection;
    private activeFeedback: Map<string, FeedbackItem> = new Map();
    private interactionCallback: InteractionCallback | null = null;

    constructor(private context: vscode.ExtensionContext) {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection(
            "eyeTrackingDebugger"
        );
        context.subscriptions.push(this.diagnosticCollection);
    }

    /**
     * Set callback for feedback interactions.
     */
    setInteractionCallback(callback: InteractionCallback): void {
        this.interactionCallback = callback;
    }

    /**
     * Render feedback items in the editor.
     */
    renderFeedback(items: FeedbackItem[]): void {
        // TODO: Implement feedback rendering

        console.log("FEEDBACK ITEMS:", items);
    }

    /**
     * Clear all rendered feedback.
     */
    clearAll(): void {
        // TODO: Implement clearing all feedback
    }

    /**
     * Clear feedback by ID.
     */
    clearFeedback(feedbackId: string): void {
        // TODO: Implement clearing specific feedback
    }

    /**
     * Show feedback as inline decoration.
     */
    showInlineDecoration(item: FeedbackItem): void {
        // TODO: Implement inline decoration
    }

    /**
     * Show feedback as hover tooltip.
     */
    showHoverTooltip(item: FeedbackItem): void {
        // TODO: Implement hover tooltip
    }

    /**
     * Show feedback as diagnostic (in Problems panel).
     */
    showAsDiagnostic(item: FeedbackItem): void {
        // TODO: Implement diagnostic display
    }

    /**
     * Show feedback as notification.
     */
    showNotification(item: FeedbackItem): void {
        // TODO: Implement notification display
    }

    /**
     * Record user interaction with feedback.
     */
    recordInteraction(
        feedbackId: string,
        interactionType: FeedbackInteraction["interaction_type"]
    ): void {
        if (this.interactionCallback) {
            const interaction: FeedbackInteraction = {
                feedback_id: feedbackId,
                interaction_type: interactionType,
                timestamp: Math.floor(Date.now() / 1000),
            };
            this.interactionCallback(interaction);
        }
    }

    /**
     * Get currently active feedback items.
     */
    getActiveFeedback(): FeedbackItem[] {
        return Array.from(this.activeFeedback.values());
    }

    /**
     * Dispose of resources.
     */
    dispose(): void {
        // TODO: Implement disposal
    }

    // --- Private Methods ---

    private getDecorationColor(priority: FeedbackPriority): string {
        // TODO: Implement color mapping
        return "yellow";
    }

    private getSeverity(priority: FeedbackPriority): vscode.DiagnosticSeverity {
        // TODO: Implement severity mapping
        return vscode.DiagnosticSeverity.Information;
    }

    private convertRange(range: CodeRange): vscode.Range {
        // TODO: Implement range conversion
        return new vscode.Range(
            range.start.line,
            range.start.character,
            range.end.line,
            range.end.character
        );
    }

    private createMarkdownMessage(item: FeedbackItem): vscode.MarkdownString {
        // TODO: Implement markdown creation
        return new vscode.MarkdownString(item.message);
    }
}
