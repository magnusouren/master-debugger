/**
 * Type definitions for messages between VS Code extension and backend.
 * These mirror the Python types in backend/types/messages.py
 */

/**
 * Types of WebSocket messages.
 */
export enum MessageType {
    // From VS Code to Backend
    CONTEXT_UPDATE = "context_update",
    CONTEXT_REQUEST = "context_request",
    FEEDBACK_INTERACTION = "feedback_interaction",

    // From Backend to VS Code
    FEEDBACK_DELIVERY = "feedback_delivery",
    STATUS_UPDATE = "status_update",
    ERROR = "error",

    // Bidirectional
    PING = "ping",
    PONG = "pong",
    CONFIG_UPDATE = "config_update",
}

/**
 * System status states.
 */
export enum SystemStatus {
    INITIALIZING = "initializing",
    READY = "ready",
    PROCESSING = "processing",
    PAUSED = "paused",
    ERROR = "error",
    DISCONNECTED = "disconnected",
}

/**
 * Feedback types.
 */
export enum FeedbackType {
    HINT = "hint",
    SUGGESTION = "suggestion",
    WARNING = "warning",
    EXPLANATION = "explanation",
    SIMPLIFICATION = "simplification",
}

/**
 * Feedback priority levels.
 */
export enum FeedbackPriority {
    LOW = "low",
    MEDIUM = "medium",
    HIGH = "high",
    CRITICAL = "critical",
}

/**
 * A position in a text document.
 */
export interface CodePosition {
    line: number;
    character: number;
}

/**
 * A range in a text document.
 */
export interface CodeRange {
    start: CodePosition;
    end: CodePosition;
}

/**
 * A diagnostic message from VS Code.
 */
export interface DiagnosticInfo {
    message: string;
    severity: string;
    range: CodeRange;
    source?: string;
    code?: string;
}

/**
 * Complete code context captured from VS Code.
 */
export interface CodeContext {
    filePath: string;
    fileContent: string;
    languageId: string;
    cursorPosition: CodePosition;
    selection?: CodeRange;
    visibleRange?: CodeRange;
    diagnostics: DiagnosticInfo[];
    workspaceFolder?: string;
    timestamp: number;
    metadata?: Record<string, unknown>;
}

/**
 * Metadata for a feedback item.
 */
export interface FeedbackMetadata {
    generatedAt: number;
    generationTimeMs: number;
    modelUsed?: string;
    cached: boolean;
    cacheKey?: string;
    feedbackId: string;
    sessionId?: string;
    extra?: Record<string, unknown>;
}

/**
 * A single feedback item.
 */
export interface FeedbackItem {
    title: string;
    message: string;
    feedbackType: FeedbackType;
    priority: FeedbackPriority;
    codeRange?: CodeRange;
    confidence: number;
    dismissible: boolean;
    actionable: boolean;
    actionLabel?: string;
    metadata: FeedbackMetadata;
}

/**
 * User interaction with feedback.
 */
export interface FeedbackInteraction {
    feedbackId: string;
    interactionType: "dismissed" | "accepted" | "clicked" | "hovered";
    timestamp: number;
    durationMs?: number;
    metadata?: Record<string, unknown>;
}

/**
 * Base WebSocket message structure.
 */
export interface WebSocketMessage {
    type: MessageType;
    timestamp: number;
    payload: Record<string, unknown>;
    messageId?: string;
}

/**
 * Feedback delivery message from backend.
 */
export interface FeedbackDeliveryPayload {
    items: FeedbackItem[];
    requestId?: string;
    triggeredBy: "reactive" | "proactive" | "manual";
    userStateScore?: number;
}

/**
 * System status update message.
 */
export interface StatusUpdatePayload {
    status: SystemStatus;
    eyeTrackerConnected: boolean;
    vscodeConnected: boolean;
    operationMode: "reactive" | "proactive";
    samplesProcessed: number;
    feedbackGenerated: number;
    errorMessage?: string;
}

/**
 * Context request from backend.
 */
export interface ContextRequestPayload {
    requestId: string;
    includeFileContent: boolean;
    includeDiagnostics: boolean;
    includeVisibleRange: boolean;
    activeFileOnly: boolean;
}
