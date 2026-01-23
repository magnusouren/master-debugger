/**
 * WebSocket client for communication with the backend.
 */
import * as vscode from "vscode";
import WebSocket from "ws";
import {
    WebSocketMessage,
    MessageType,
    CodeContext,
    FeedbackInteraction,
} from "./types";

export type MessageHandler = (message: WebSocketMessage) => void;

export class WebSocketClient {
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000;
    private messageHandlers: Map<MessageType, MessageHandler[]> = new Map();
    private isConnecting = false;
    private shouldReconnect = true;
    private reconnectTimer: NodeJS.Timeout | null = null;
    private pingInterval: NodeJS.Timeout | null = null;

    constructor(
        private host: string,
        private port: number
    ) {}

    /**
     * Connect to the backend WebSocket server.
     */
    async connect(): Promise<boolean> {
        if (this.isConnecting) {
            console.log("Connection already in progress");
            return false;
        }

        if (this.isConnected()) {
            console.log("Already connected");
            return true;
        }

        this.isConnecting = true;
        this.shouldReconnect = true;

        return new Promise<boolean>((resolve) => {
            const url = `ws://${this.host}:${this.port}`;
            console.log(`Connecting to WebSocket server at ${url}`);

            try {
                this.ws = new WebSocket(url);

                const connectionTimeout = setTimeout(() => {
                    if (this.isConnecting) {
                        console.log("Connection timeout");
                        this.ws?.close();
                        this.isConnecting = false;
                        resolve(false);
                    }
                }, 10000); // 10 second timeout

                this.ws.on("open", () => {
                    clearTimeout(connectionTimeout);
                    this.handleOpen();
                    this.isConnecting = false;
                    this.reconnectAttempts = 0;
                    resolve(true);
                });

                this.ws.on("close", (code, reason) => {
                    clearTimeout(connectionTimeout);
                    this.handleClose(code, reason.toString());
                    if (this.isConnecting) {
                        this.isConnecting = false;
                        resolve(false);
                    }
                });

                this.ws.on("error", (error) => {
                    clearTimeout(connectionTimeout);
                    this.handleError(error);
                    if (this.isConnecting) {
                        this.isConnecting = false;
                        resolve(false);
                    }
                });

                this.ws.on("message", (data) => {
                    this.handleMessage(data);
                });

            } catch (error) {
                console.error("Failed to create WebSocket:", error);
                this.isConnecting = false;
                resolve(false);
            }
        });
    }

    /**
     * Disconnect from the backend.
     */
    disconnect(): void {
        this.shouldReconnect = false;
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }

        if (this.ws) {
            this.ws.close(1000, "Client disconnect");
            this.ws = null;
        }

        this.reconnectAttempts = 0;
        console.log("Disconnected from WebSocket server");
    }

    /**
     * Check if connected to the backend.
     */
    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Send a message to the backend.
     */
    send(message: WebSocketMessage): boolean {
        if (!this.isConnected()) {
            console.warn("Cannot send message: not connected");
            return false;
        }

        try {
            const data = JSON.stringify(message);
            this.ws!.send(data);
            return true;
        } catch (error) {
            console.error("Failed to send message:", error);
            return false;
        }
    }

    /**
     * Send code context to the backend.
     */
    sendContextUpdate(context: CodeContext): boolean {
        const message = this.createMessage(MessageType.CONTEXT_UPDATE, {
            ...context,
        } as unknown as Record<string, unknown>);
        return this.send(message);
    }

    /**
     * Send feedback interaction to the backend.
     */
    sendFeedbackInteraction(interaction: FeedbackInteraction): boolean {
        const message = this.createMessage(MessageType.FEEDBACK_INTERACTION, {
            ...interaction,
        } as unknown as Record<string, unknown>);
        return this.send(message);
    }

    /**
     * Register a handler for a specific message type.
     */
    onMessage(type: MessageType, handler: MessageHandler): void {
        const handlers = this.messageHandlers.get(type) || [];
        handlers.push(handler);
        this.messageHandlers.set(type, handlers);
    }

    /**
     * Remove a handler for a specific message type.
     */
    offMessage(type: MessageType, handler: MessageHandler): void {
        const handlers = this.messageHandlers.get(type);
        if (handlers) {
            const index = handlers.indexOf(handler);
            if (index !== -1) {
                handlers.splice(index, 1);
            }
        }
    }

    /**
     * Update connection settings.
     */
    updateSettings(host: string, port: number): void {
        const wasConnected = this.isConnected();
        
        if (host !== this.host || port !== this.port) {
            this.host = host;
            this.port = port;

            // Reconnect if we were connected
            if (wasConnected) {
                this.disconnect();
                this.connect();
            }
        }
    }

    // --- Private Methods ---

    private handleOpen(): void {
        console.log("WebSocket connection established");
        
        // Start ping interval to keep connection alive
        this.pingInterval = setInterval(() => {
            if (this.isConnected()) {
                this.send(this.createMessage(MessageType.PING, {}));
            }
        }, 30000); // Ping every 30 seconds
    }

    private handleClose(code: number, reason: string): void {
        console.log(`WebSocket closed: code=${code}, reason=${reason}`);
        
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }

        this.ws = null;

        if (this.shouldReconnect) {
            this.scheduleReconnect();
        }
    }

    private handleError(error: Error): void {
        console.error("WebSocket error:", error.message);
        vscode.window.showWarningMessage(
            `Eye Tracking connection error: ${error.message}`
        );
    }

    private handleMessage(data: WebSocket.Data): void {
        const dataStr = data.toString();
        const message = this.parseMessage(dataStr);
        
        if (message) {
            // Handle pong silently
            if (message.type === MessageType.PONG) {
                return;
            }
            
            this.dispatchMessage(message);
        }
    }

    private parseMessage(data: string): WebSocketMessage | null {
        try {
            const parsed = JSON.parse(data) as WebSocketMessage;
            
            // Validate required fields
            if (!parsed.type || typeof parsed.timestamp !== "number") {
                console.warn("Invalid message format:", data);
                return null;
            }
            
            return parsed;
        } catch (error) {
            console.error("Failed to parse message:", error);
            return null;
        }
    }

    private dispatchMessage(message: WebSocketMessage): void {
        const handlers = this.messageHandlers.get(message.type);
        
        if (handlers && handlers.length > 0) {
            for (const handler of handlers) {
                try {
                    handler(message);
                } catch (error) {
                    console.error(`Handler error for ${message.type}:`, error);
                }
            }
        } else {
            console.log(`No handler for message type: ${message.type}`);
        }
    }

    private scheduleReconnect(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log("Max reconnect attempts reached");
            vscode.window.showErrorMessage(
                "Failed to reconnect to Eye Tracking backend after multiple attempts"
            );
            return;
        }

        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
        this.reconnectAttempts++;

        console.log(
            `Scheduling reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`
        );

        this.reconnectTimer = setTimeout(async () => {
            console.log(`Reconnect attempt ${this.reconnectAttempts}`);
            const connected = await this.connect();
            
            if (connected) {
                vscode.window.showInformationMessage(
                    "Reconnected to Eye Tracking backend"
                );
            }
        }, delay);
    }

    private createMessage(
        type: MessageType,
        payload: Record<string, unknown>
    ): WebSocketMessage {
        return {
            type,
            timestamp: Date.now() / 1000,
            payload,
            messageId: this.generateMessageId(),
        };
    }

    private generateMessageId(): string {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}
