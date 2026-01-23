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
        // TODO: Implement context collection
        return null;
    }

    /**
     * Collect context from a specific editor.
     */
    collectContextFromEditor(editor: vscode.TextEditor): CodeContext {
        // TODO: Implement context collection from specific editor
        return this.createEmptyContext();
    }

    /**
     * Get the current cursor position.
     */
    getCursorPosition(editor: vscode.TextEditor): CodePosition {
        // TODO: Implement cursor position extraction
        return { line: 0, character: 0 };
    }

    /**
     * Get the current selection range.
     */
    getSelection(editor: vscode.TextEditor): CodeRange | undefined {
        // TODO: Implement selection extraction
        return undefined;
    }

    /**
     * Get the visible range in the editor.
     */
    getVisibleRange(editor: vscode.TextEditor): CodeRange | undefined {
        // TODO: Implement visible range extraction
        return undefined;
    }

    /**
     * Get diagnostics for the current document.
     */
    getDiagnostics(document: vscode.TextDocument): DiagnosticInfo[] {
        // TODO: Implement diagnostics extraction
        return [];
    }

    /**
     * Convert VS Code position to CodePosition.
     */
    private convertPosition(position: vscode.Position): CodePosition {
        // TODO: Implement position conversion
        return { line: position.line, character: position.character };
    }

    /**
     * Convert VS Code range to CodeRange.
     */
    private convertRange(range: vscode.Range): CodeRange {
        // TODO: Implement range conversion
        return {
            start: this.convertPosition(range.start),
            end: this.convertPosition(range.end),
        };
    }

    /**
     * Convert VS Code diagnostic severity to string.
     */
    private convertSeverity(severity: vscode.DiagnosticSeverity): string {
        // TODO: Implement severity conversion
        return "info";
    }

    /**
     * Create an empty context object.
     */
    private createEmptyContext(): CodeContext {
        return {
            filePath: "",
            fileContent: "",
            languageId: "",
            cursorPosition: { line: 0, character: 0 },
            diagnostics: [],
            timestamp: Date.now() / 1000,
        };
    }
}
