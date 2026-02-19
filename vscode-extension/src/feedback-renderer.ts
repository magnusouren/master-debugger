/**
 * Feedback renderer - displays feedback in the VS Code editor.
 */
import * as vscode from 'vscode';
import {
    FeedbackItem,
    FeedbackType,
    FeedbackPriority,
    CodeRange,
    FeedbackInteraction,
    InteractionType,
} from './types';

export type InteractionCallback = (interaction: FeedbackInteraction) => void;

export class FeedbackRenderer {
    private decorationType: vscode.TextEditorDecorationType | null = null;
    private diagnosticCollection: vscode.DiagnosticCollection;
    private activeFeedback: Map<string, FeedbackItem> = new Map();
    private interactionCallback: InteractionCallback | null = null;

    constructor(private context: vscode.ExtensionContext) {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection(
            'eyeTrackingDebugger',
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
     * Add feedback to the renderer, but don't display it yet.
     */
    addFeedback(items: FeedbackItem[]): void {
        items.forEach((item) => {
            this.activeFeedback.set(item.metadata.feedback_id, item);
            this.showNotification(item);
        });
    }

    /**
     * Activate existing feedback items (e.g. show decorations, diagnostics, etc.).
     */
    activateFeedback(feedbackId: string): void {
        const item = this.activeFeedback.get(feedbackId);
        if (!item) return;

        // For simplicity, we'll show all feedback as diagnostics and inline decorations
        this.showAsDiagnostic(item);
        // this.showInlineDecoration(item);
        // this.highlightCodeRange(item);
    }

    /**
     * Clear all rendered feedback.
     */
    clearAll(): void {
        this.activeFeedback.clear();
        this.diagnosticCollection.clear();
        if (this.decorationType) {
            this.decorationType.dispose();
            this.decorationType = null;
        }
    }

    /**
     * Clear feedback by ID.
     */
    clearFeedback(feedbackId: string): void {
        this.activeFeedback.delete(feedbackId);
        const diagnostics = this.diagnosticCollection.get(
            vscode.window.activeTextEditor?.document.uri ||
                vscode.Uri.parse(''),
        );
        if (diagnostics) {
            const updatedDiagnostics = diagnostics.filter(
                (diag) => diag.code !== feedbackId,
            );
            this.diagnosticCollection.set(
                vscode.window.activeTextEditor?.document.uri ||
                    vscode.Uri.parse(''),
                updatedDiagnostics,
            );
        }
    }

    /**
     * Show feedback as inline decoration.
     */
    showInlineDecoration(item: FeedbackItem): void {
        if (!item.code_range) return;

        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const range = this.convertRange(item.code_range, editor.document);
        const decorationColor = this.getDecorationColor(item.priority);

        if (!this.decorationType) {
            this.decorationType = vscode.window.createTextEditorDecorationType({
                backgroundColor: decorationColor,
                isWholeLine: false,
            });
        }

        editor.setDecorations(this.decorationType, [range]);
    }

    /**
     * Show feedback as hover tooltip.
     */
    showHoverTooltip(item: FeedbackItem): void {
        if (!item.code_range) return;

        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const range = this.convertRange(item.code_range, editor.document);
        const markdownMessage = this.createMarkdownMessage(item);

        const hoverProvider: vscode.HoverProvider = {
            provideHover() {
                return new vscode.Hover(markdownMessage, range);
            },
        };

        const selector: vscode.DocumentSelector = [
            { scheme: 'file', language: editor.document.languageId },
        ];

        this.context.subscriptions.push(
            vscode.languages.registerHoverProvider(selector, hoverProvider),
        );
    }

    /**
     * Show feedback as diagnostic (in Problems panel).
     */
    showAsDiagnostic(item: FeedbackItem): void {
        if (!item.code_range) return;

        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const range = this.convertRange(item.code_range, editor.document);
        const severity = this.getSeverity(item.priority);
        const diagnostic = new vscode.Diagnostic(range, item.message, severity);
        diagnostic.code = item.metadata.feedback_id;

        this.diagnosticCollection.set(editor.document.uri, [diagnostic]);
    }

    /**
     * Show feedback as notification.
     */
    showNotification(item: FeedbackItem): void {
        vscode.window
            .showInformationMessage(
                // `${item.title}: ${item.message}`,
                'Feedback Available',
                'View Details',
                'Dismiss',
            )
            .then((selection) => {
                if (selection === 'View Details') {
                    this.interactionCallback?.({
                        feedback_id: item.metadata.feedback_id,
                        interaction_type: 'accepted',
                        timestamp: Math.floor(Date.now() / 1000),
                    });
                    this.showHoverTooltip(item);
                } else if (selection === 'Dismiss') {
                    this.interactionCallback?.({
                        feedback_id: item.metadata.feedback_id,
                        interaction_type: 'rejected',
                        timestamp: Math.floor(Date.now() / 1000),
                    });
                    this.clearFeedback(item.metadata.feedback_id);
                }
            });
    }

    /**
     * Record user interaction with feedback.
     */
    recordInteraction(
        feedbackId: string,
        interactionType: InteractionType,
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
        this.clearAll();
        this.diagnosticCollection.dispose();
    }

    // --- Private Methods ---

    private getDecorationColor(priority: FeedbackPriority): string {
        // TODO: Implement color mapping
        return 'yellow';
    }

    private getSeverity(priority: FeedbackPriority): vscode.DiagnosticSeverity {
        // TODO: Implement severity mapping
        return vscode.DiagnosticSeverity.Information;
    }

    private convertRange(
        range: CodeRange,
        document: vscode.TextDocument,
    ): vscode.Range {
        const clamp = (n: number, min: number, max: number) =>
            Math.max(min, Math.min(max, n));

        const startLine = clamp(range.start.line, 0, document.lineCount - 1);
        const endLine = clamp(range.end.line, 0, document.lineCount - 1);

        const start = document.lineAt(startLine).range.start;
        const end = document.lineAt(endLine).range.end;

        return new vscode.Range(start, end);
    }

    private createMarkdownMessage(item: FeedbackItem): vscode.MarkdownString {
        // TODO: Implement markdown creation
        return new vscode.MarkdownString(item.message);
    }
}
