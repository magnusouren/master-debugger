/**
 * Feedback renderer - displays feedback in the VS Code editor.
 */
import * as vscode from 'vscode';
import {
    FeedbackItem,
    CodeRange,
    FeedbackInteraction,
    InteractionType,
} from './types';

export type InteractionCallback = (interaction: FeedbackInteraction) => void;

/** Command ID for dismissing feedback */
export const DISMISS_FEEDBACK_COMMAND = 'eyeTrackingDebugger.dismissFeedback';

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

        // Register dismiss command
        const dismissCommand = vscode.commands.registerCommand(
            DISMISS_FEEDBACK_COMMAND,
            (feedbackId: string) => (
                this.recordInteraction(feedbackId, 'dismissed'),
                this.removeHighlightById(feedbackId)
            ),
        );
        context.subscriptions.push(dismissCommand);

        // Register code action provider for dismiss quick fix
        const codeActionProvider = new FeedbackCodeActionProvider(
            this.diagnosticCollection,
        );
        context.subscriptions.push(
            vscode.languages.registerCodeActionsProvider(
                { scheme: 'file' },
                codeActionProvider,
                { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] },
            ),
        );
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
        });
    }

    /**
     * Activate existing feedback items (e.g. show decorations, diagnostics, etc.).
     */
    highlightFeedback(feedbackId: string): void {
        const item = this.activeFeedback.get(feedbackId);
        if (!item) return;

        this.showAsDiagnostic(item);
    }

    /**
     * Clear feedback by ID.
     */
    removeHighlightById(feedbackId: string): void {
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
     * Show feedback as diagnostic (in Problems panel).
     */
    private showAsDiagnostic(item: FeedbackItem): void {
        if (!item.code_range) return;

        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const range = this.convertRange(item.code_range, editor.document);
        const severity = vscode.DiagnosticSeverity.Information;
        const diagnostic = new vscode.Diagnostic(range, item.message, severity);
        diagnostic.code = item.metadata.feedback_id;

        // Get existing diagnostics and filter out any with the same feedback_id
        const existingDiagnostics =
            this.diagnosticCollection.get(editor.document.uri) || [];
        const filteredDiagnostics = existingDiagnostics.filter(
            (d) => d.code !== item.metadata.feedback_id,
        );

        // Merge with the new diagnostic
        this.diagnosticCollection.set(editor.document.uri, [
            ...filteredDiagnostics,
            diagnostic,
        ]);
    }

    /**
     * Record user interaction with feedback.
     */
    private recordInteraction(
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

    private convertRange(
        range: CodeRange,
        document: vscode.TextDocument,
    ): vscode.Range {
        const clamp = (n: number, min: number, max: number) =>
            Math.max(min, Math.min(max, n));

        const startLine = clamp(range.start.line, 0, document.lineCount - 1);
        const endLine = clamp(range.end.line, 0, document.lineCount - 1);

        const startLineLength = document.lineAt(startLine).text.length;
        const endLineLength = document.lineAt(endLine).text.length;

        const startChar = clamp(range.start.character, 0, startLineLength);
        const endChar = clamp(range.end.character, 0, endLineLength);

        return new vscode.Range(startLine, startChar, endLine, endChar);
    }

    /**
     * Dispose of resources.
     */
    dispose(): void {
        this.clearAll();
        this.diagnosticCollection.dispose();
    }
}

/**
 * Code action provider that offers "Dismiss Feedback" quick fix for feedback diagnostics.
 */
class FeedbackCodeActionProvider implements vscode.CodeActionProvider {
    constructor(private diagnosticCollection: vscode.DiagnosticCollection) {}

    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
    ): vscode.CodeAction[] {
        const actions: vscode.CodeAction[] = [];

        // Get our diagnostics for this document
        const ourDiagnostics =
            this.diagnosticCollection.get(document.uri) || [];

        // Find diagnostics that intersect with the current range
        for (const diagnostic of context.diagnostics) {
            // Check if this diagnostic belongs to our collection
            const isOurDiagnostic = ourDiagnostics.some(
                (d) =>
                    d.code === diagnostic.code &&
                    d.range.isEqual(diagnostic.range),
            );

            if (isOurDiagnostic && diagnostic.code) {
                const feedbackId = String(diagnostic.code);
                const action = new vscode.CodeAction(
                    'Dismiss Feedback',
                    vscode.CodeActionKind.QuickFix,
                );
                action.command = {
                    command: DISMISS_FEEDBACK_COMMAND,
                    title: 'Dismiss Feedback',
                    arguments: [feedbackId],
                };
                action.diagnostics = [diagnostic];
                action.isPreferred = false;
                actions.push(action);
            }
        }

        return actions;
    }
}
