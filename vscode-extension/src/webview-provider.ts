/**
 * WebviewViewProvider for the Eye Tracking Debugger feedback panel.
 * 
 * This provider manages the webview that displays the React-based UI.
 */
import * as vscode from "vscode";
import { FeedbackItem, SystemStatusMessage } from "./types";

// Set to true during development to load from Vite dev server
const DEV_MODE = false;
const DEV_SERVER_URL = "http://localhost:5173";

export class FeedbackViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = "eyeTrackingDebugger.feedbackView";

    private _view?: vscode.WebviewView;
    private _extensionUri: vscode.Uri;

    // Callbacks for handling webview messages
    private _onConnect?: () => void;
    private _onDisconnect?: () => void;
    private _onToggleMode?: () => void;
    private _onClearFeedback?: () => void;
    private _onTriggerFeedback?: () => void;
    private _onConnectEyeTracker?: () => void;
    private _onDisconnectEyeTracker?: () => void;
    private _onFeedbackInteraction?: (feedbackId: string, interactionType: "dismissed" | "accepted") => void;

    constructor(extensionUri: vscode.Uri) {
        this._extensionUri = extensionUri;
    }

    /**
     * Set up callbacks for webview actions
     */
    public setCallbacks(callbacks: {
        onConnect?: () => void;
        onDisconnect?: () => void;
        onToggleMode?: () => void;
        onClearFeedback?: () => void;
        onTriggerFeedback?: () => void;
        onConnectEyeTracker?: () => void;
        onDisconnectEyeTracker?: () => void;
        onFeedbackInteraction?: (feedbackId: string, interactionType: "dismissed" | "accepted") => void;
    }): void {
        this._onConnect = callbacks.onConnect;
        this._onDisconnect = callbacks.onDisconnect;
        this._onToggleMode = callbacks.onToggleMode;
        this._onClearFeedback = callbacks.onClearFeedback;
        this._onTriggerFeedback = callbacks.onTriggerFeedback;
        this._onConnectEyeTracker = callbacks.onConnectEyeTracker;
        this._onDisconnectEyeTracker = callbacks.onDisconnectEyeTracker;
        this._onFeedbackInteraction = callbacks.onFeedbackInteraction;
    }

    /**
     * Called when the view is resolved (becomes visible).
     */
    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this._extensionUri, "webview-ui", "build"),
            ],
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage((message) => {
            this._handleWebviewMessage(message);
        });

        // When view becomes visible, sync state
        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible) {
                // Webview became visible - extension can send current state
                console.log("Webview became visible");
            }
        });
    }

    /**
     * Update the connection status in the webview
     */
    public updateConnectionStatus(connected: boolean): void {
        this._postMessage({
            type: "connectionStatus",
            payload: { connected },
        });
    }

    /**
     * Update the system status in the webview
     */
    public updateStatus(status: SystemStatusMessage): void {
        this._postMessage({
            type: "statusUpdate",
            payload: status,
        });
    }

    /**
     * Update feedback items in the webview
     */
    public updateFeedback(items: FeedbackItem[]): void {
        this._postMessage({
            type: "feedbackUpdate",
            payload: { items },
        });
    }

    /**
     * Clear all feedback in the webview
     */
    public clearFeedback(): void {
        this._postMessage({
            type: "clearFeedback",
            payload: {},
        });
    }

    /**
     * Handle messages from the webview
     */
    private _handleWebviewMessage(message: { type: string; payload?: unknown }): void {
        switch (message.type) {
            case "ready":
                // Webview is ready - send initial state
                console.log("Webview ready, sending initial state");
                break;
            case "connect":
                this._onConnect?.();
                break;
            case "disconnect":
                this._onDisconnect?.();
                break;
            case "toggleMode":
                this._onToggleMode?.();
                break;
            case "clearFeedback":
                this._onClearFeedback?.();
                break;
            case "triggerFeedback":
                this._onTriggerFeedback?.();
                break;
            case "connectEyeTracker":
                this._onConnectEyeTracker?.();
                break;
            case "disconnectEyeTracker":
                this._onDisconnectEyeTracker?.();
                break;
            case "feedbackInteraction":
                const payload = message.payload as { feedbackId: string; interactionType: "dismissed" | "accepted" };
                this._onFeedbackInteraction?.(payload.feedbackId, payload.interactionType);
                break;
            default:
                console.log("Unknown webview message:", message.type);
        }
    }

    /**
     * Post a message to the webview
     */
    private _postMessage(message: { type: string; payload: unknown }): void {
        if (this._view) {
            this._view.webview.postMessage(message);
        }
    }

    /**
     * Generate the HTML content for the webview
     */
    private _getHtmlForWebview(webview: vscode.Webview): string {
        // Use a nonce for security
        const nonce = this._getNonce();

        // In dev mode, load from Vite dev server for hot reload
        if (DEV_MODE) {
            return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eye Tracking Feedback</title>
</head>
<body>
    <div id="root"></div>
    <script type="module">
        import RefreshRuntime from "${DEV_SERVER_URL}/@react-refresh"
        RefreshRuntime.injectIntoGlobalHook(window)
        window.$RefreshReg$ = () => {}
        window.$RefreshSig$ = () => (type) => type
        window.__vite_plugin_react_preamble_installed__ = true
    </script>
    <script type="module" src="${DEV_SERVER_URL}/src/main.tsx"></script>
</body>
</html>`;
        }

        // Production mode: load from built assets
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, "webview-ui", "build", "assets", "main.js")
        );
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, "webview-ui", "build", "assets", "main.css")
        );

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <link href="${styleUri}" rel="stylesheet">
    <title>Eye Tracking Feedback</title>
</head>
<body>
    <div id="root"></div>
    <script type="module" nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
    }

    /**
     * Generate a nonce for CSP
     */
    private _getNonce(): string {
        let text = "";
        const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        for (let i = 0; i < 32; i++) {
            text += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        return text;
    }
}
