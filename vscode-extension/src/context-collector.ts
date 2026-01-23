/**
 * Context collector - captures editor state for the backend.
 */
import * as vscode from "vscode";
import {
    CodeContext,
    CodePosition,
    CodeRange,
    DiagnosticInfo,
} from "./types";

export class ContextCollector {
    constructor() {}

    /**
     * Collect current code context from the active editor.
     */
    collectContext(): CodeContext | null {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            return this.collectContextFromEditor(editor);
        }
        return null;
    }

    /**
     * Collect context from a specific editor.
     */
    collectContextFromEditor(editor: vscode.TextEditor): CodeContext | null {
        if (!editor) {
            return null;
        }

        const document = editor.document;

        if (document) {
            const file_path = document.uri.fsPath;
            const file_content = document.getText();
            const language_id = document.languageId;
            const cursor_position = this.getCursorPosition(editor);
            const selection = this.getSelection(editor);
            const visible_range = this.getVisibleRange(editor);
            const diagnostics = this.getDiagnostics(document);

            return {
                file_path,
                file_content,
                language_id,
                cursor_position,
                selection,
                visible_range,
                diagnostics,
                timestamp: Date.now(),
            };
        }


        return this.createEmptyContext();
    }

    /**
     * Get the current cursor position.
     */
    getCursorPosition(editor: vscode.TextEditor): CodePosition {
        if (editor.selection) {
            const position = editor.selection.active;
            return this.convertPosition(position);
        }

        return { line: 0, character: 0 };
    }

    /**
     * Get the current selection range.
     */
    getSelection(editor: vscode.TextEditor): CodeRange | undefined {
        if (editor.selection && !editor.selection.isEmpty) {
            return this.convertRange(editor.selection);
        }

        return undefined;
    }

    /**
     * Get the visible range in the editor.
     */
    getVisibleRange(editor: vscode.TextEditor): CodeRange | undefined {
        const visibleRanges = editor.visibleRanges;
        if (visibleRanges && visibleRanges.length > 0) {
            return this.convertRange(visibleRanges[0]);
        }

        return undefined;
    }

    /**
     * Get diagnostics for the current document.
     */
    getDiagnostics(document: vscode.TextDocument): DiagnosticInfo[] {
        const diagnostics = vscode.languages.getDiagnostics(document.uri);
        if (diagnostics && diagnostics.length > 0) {
            return diagnostics.map((diag) => ({
                message: diag.message,
                severity: this.convertSeverity(diag.severity),
                range: this.convertRange(diag.range),
                source: diag.source,
                code: diag.code ? diag.code.toString() : undefined,
            }));
        }

        return [];
    }

    /**
     * Convert VS Code position to CodePosition.
     */
    private convertPosition(position: vscode.Position): CodePosition {
        return { line: position.line, character: position.character };
    }

    /**
     * Convert VS Code range to CodeRange.
     */
    private convertRange(range: vscode.Range): CodeRange {
        return {
            start: this.convertPosition(range.start),
            end: this.convertPosition(range.end),
        };
    }

    /**
     * Convert VS Code diagnostic severity to string.
     */
    private convertSeverity(severity: vscode.DiagnosticSeverity): string {
        switch (severity) {
            case vscode.DiagnosticSeverity.Error:
                return "error";
            case vscode.DiagnosticSeverity.Warning:
                return "warning";
            case vscode.DiagnosticSeverity.Information:
                return "info";
            case vscode.DiagnosticSeverity.Hint:
                return "hint";
        }
    }

    /**
     * Create an empty context object.
     */
    private createEmptyContext(): CodeContext {
        return {
            file_path: "",
            file_content: "",
            language_id: "",
            cursor_position: { line: 0, character: 0 },
            diagnostics: [],
            timestamp: Date.now(),
        };
    }
}
