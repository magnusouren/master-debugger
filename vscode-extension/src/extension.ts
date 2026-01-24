/**
 * Eye Tracking Debugger Extension
 * 
 * Main entry point for the VS Code extension.
 * Handles extension activation, command registration, and lifecycle management.
 */
import * as vscode from "vscode";
import { WebSocketClient } from "./websocket-client";
import { ContextCollector } from "./context-collector";
import { FeedbackRenderer } from "./feedback-renderer";
import { StatusBarManager } from "./status-bar";
import {
    MessageType,
    FeedbackDeliveryPayload,
    StatusUpdatePayload,
    ContextRequestPayload,
    SystemStatus,
} from "./types";

let wsClient: WebSocketClient | null = null;
let contextCollector: ContextCollector | null = null;
let feedbackRenderer: FeedbackRenderer | null = null;
let statusBar: StatusBarManager | null = null;

// Debounce timer for context updates
let contextUpdateTimer: NodeJS.Timeout | null = null;
const CONTEXT_UPDATE_DEBOUNCE_MS = 500;

/**
 * Extension activation.
 */
export function activate(context: vscode.ExtensionContext): void {
    console.log("Eye Tracking Debugger extension is now active");

    // Initialize components
    initializeComponents(context);

    // Register commands
    registerCommands(context);

    // Set up event listeners
    setupEventListeners(context);

    // Auto-connect if configured
    const config = vscode.workspace.getConfiguration("eyeTrackingDebugger");
    if (config.get<boolean>("autoConnect")) {
        connectToBackend();
    }
}

/**
 * Extension deactivation.
 */
export function deactivate(): void {
    // TODO: Implement cleanup
    wsClient?.disconnect();
    feedbackRenderer?.dispose();
    statusBar?.dispose();
}

/**
 * Initialize all extension components.
 */
function initializeComponents(context: vscode.ExtensionContext): void {
    // TODO: Implement component initialization
    const config = vscode.workspace.getConfiguration("eyeTrackingDebugger");
    const host = config.get<string>("backendHost") || "localhost";
    const port = config.get<number>("websocketPort") || 8765;

    wsClient = new WebSocketClient(host, port);
    contextCollector = new ContextCollector();
    feedbackRenderer = new FeedbackRenderer(context);
    statusBar = new StatusBarManager(context);

    // Set up message handlers
    setupMessageHandlers();
}

/**
 * Register extension commands.
 */
function registerCommands(context: vscode.ExtensionContext): void {
    // Connect command
    context.subscriptions.push(
        vscode.commands.registerCommand(
            "eyeTrackingDebugger.connect",
            connectToBackend
        )
    );

    // Disconnect command
    context.subscriptions.push(
        vscode.commands.registerCommand(
            "eyeTrackingDebugger.disconnect",
            disconnectFromBackend
        )
    );

    // Toggle mode command
    context.subscriptions.push(
        vscode.commands.registerCommand(
            "eyeTrackingDebugger.toggleMode",
            toggleMode
        )
    );

    // Show status command
    context.subscriptions.push(
        vscode.commands.registerCommand(
            "eyeTrackingDebugger.showStatus",
            showStatus
        )
    );

    // Clear feedback command
    context.subscriptions.push(
        vscode.commands.registerCommand(
            "eyeTrackingDebugger.clearFeedback",
            clearFeedback
        )
    );
}

/**
 * Set up event listeners for editor changes.
 */
function setupEventListeners(context: vscode.ExtensionContext): void {
    // TODO: Implement event listeners

    // Listen for active editor changes
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(onActiveEditorChanged)
    );

    // Listen for document changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(onDocumentChanged)
    );

    // Listen for selection changes
    context.subscriptions.push(
        vscode.window.onDidChangeTextEditorSelection(onSelectionChanged)
    );

    // Listen for configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(onConfigurationChanged)
    );
}

/**
 * Set up WebSocket message handlers.
 */
function setupMessageHandlers(): void {
    // TODO: Implement message handler setup
    if (!wsClient) return;

    wsClient.onMessage(MessageType.FEEDBACK_DELIVERY, handleFeedbackDelivery);
    wsClient.onMessage(MessageType.STATUS_UPDATE, handleStatusUpdate);
    wsClient.onMessage(MessageType.CONTEXT_REQUEST, handleContextRequest);
    wsClient.onMessage(MessageType.ERROR, handleError);
}

// --- Command Handlers ---

async function connectToBackend(): Promise<void> {
    // TODO: Implement backend connection
    if (!wsClient) return;

    const connected = await wsClient.connect();
    if (connected) {
        vscode.window.showInformationMessage(
            "Connected to Eye Tracking backend"
        );
        statusBar?.setConnected(true);

        // Send initial context immediately
        sendContextUpdate();
    } else {
        vscode.window.showErrorMessage(
            "Failed to connect to Eye Tracking backend"
        );
    }
}

function disconnectFromBackend(): void {
    // TODO: Implement backend disconnection
    wsClient?.disconnect();
    statusBar?.setConnected(false);
    vscode.window.showInformationMessage(
        "Disconnected from Eye Tracking backend"
    );
}

async function toggleMode(): Promise<void> {
    // TODO: Implement mode toggling
    vscode.window.showInformationMessage("Mode toggle not yet implemented");
}

function showStatus(): void {
    // TODO: Implement status display
    vscode.window.showInformationMessage("Status display not yet implemented");
}

function clearFeedback(): void {
    // TODO: Implement feedback clearing
    feedbackRenderer?.clearAll();
    vscode.window.showInformationMessage("Feedback cleared");
}

// --- Event Handlers ---

function onActiveEditorChanged(editor: vscode.TextEditor | undefined): void {
    if (editor) {
        console.log(`Active editor changed: ${editor.document.uri.toString()}`);
        // Send context immediately when switching files
        sendContextUpdate();
    } else {
        console.log("No active editor");
    }
}

function onDocumentChanged(event: vscode.TextDocumentChangeEvent): void {
    // Only track changes in the active editor
    const activeEditor = vscode.window.activeTextEditor;
    if (activeEditor && event.document === activeEditor.document) {
        console.log(`Document changed: ${event.document.uri.toString()}`);
        // Debounce context updates during typing
        scheduleContextUpdate();
    }
}

function onSelectionChanged(event: vscode.TextEditorSelectionChangeEvent): void {
    // TODO - only track selection changes in the active editor, dont send the whole file again
    console.log(`Selection changed in: ${event.textEditor.document.uri.toString()}`);
    // Debounce selection changes (cursor movement)
    scheduleContextUpdate();
}

/**
 * Schedule a debounced context update.
 */
function scheduleContextUpdate(): void {
    if (contextUpdateTimer) {
        clearTimeout(contextUpdateTimer);
    }
    contextUpdateTimer = setTimeout(() => {
        sendContextUpdate();
    }, CONTEXT_UPDATE_DEBOUNCE_MS);
}

/**
 * Send current context to the backend.
 */
function sendContextUpdate(): void {
    if (!wsClient?.isConnected()) {
        console.log("WebSocket not connected, skipping context update");
        return;
    }

    const context = contextCollector?.collectContext();
    if (context) {
        console.log(`Sending context update for: ${context.file_path}`);
        wsClient.sendContextUpdate(context);
    }
}

function onConfigurationChanged(event: vscode.ConfigurationChangeEvent): void {
    if (event.affectsConfiguration("eyeTrackingDebugger")) {
        updateConfiguration();
    }
}

function updateConfiguration(): void {
    const config = vscode.workspace.getConfiguration("eyeTrackingDebugger");
    const host = config.get<string>("backendHost") || "localhost";
    const port = config.get<number>("websocketPort") || 8765;
    wsClient?.updateSettings(host, port);
}

// --- Message Handlers ---

function handleFeedbackDelivery(message: { payload: Record<string, unknown> }): void {
    console.log("Received feedback delivery message");

    const payload = message.payload as unknown as FeedbackDeliveryPayload;
    feedbackRenderer?.renderFeedback(payload.items);
}

function handleStatusUpdate(message: { payload: Record<string, unknown> }): void {
    // TODO: Implement status update handling
    const payload = message.payload as unknown as StatusUpdatePayload;
    statusBar?.setStatus(payload.status);
}

function handleContextRequest(message: { payload: Record<string, unknown> }): void {
    // TODO: Implement context request handling
    const payload = message.payload as unknown as ContextRequestPayload;
    const context = contextCollector?.collectContext();
    if (context && wsClient) {
        wsClient.sendContextUpdate(context);
    }
}

function handleError(message: { payload: Record<string, unknown> }): void {
    // TODO: Implement error handling
    const errorMessage = message.payload["message"] as string || "Unknown error";
    vscode.window.showErrorMessage(`Eye Tracking Error: ${errorMessage}`);
}
