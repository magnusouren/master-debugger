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
    content?: string;
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
 *  */
export interface CodeContext {
    file_path: string;
    file_content?: string;
    language_id: string;
    cursor_position: CodePosition;
    selection?: CodeRange;
    visible_range?: CodeRange;
    diagnostics: DiagnosticInfo[];
    workspace_folder?: string;
    timestamp: number;
    metadata?: Record<string, unknown>;
}

/**
 * Metadata for a feedback item.
 */
export interface FeedbackMetadata {
    generated_at: number;
    generation_time_ms: number;
    model_used?: string;
    cached: boolean;
    cache_key?: string;
    feedback_id: string;
    session_id?: string;
    extra?: Record<string, unknown>;
}

/**
 * A single feedback item.
 */
export interface FeedbackItem {
    title: string;
    message: string;
    feedback_type: FeedbackType;
    priority: FeedbackPriority;
    code_range?: CodeRange;
    confidence: number;
    dismissible: boolean;
    actionable: boolean;
    action_label?: string;
    metadata: FeedbackMetadata;
}

/**
 * User interaction with feedback.
 */
export interface FeedbackInteraction {
    feedback_id: string;
    interaction_type: "dismissed" | "accepted" | "clicked" | "hovered";
    timestamp: number;
    duration_ms?: number;
    metadata?: Record<string, unknown>;
}

/**
 * Base WebSocket message structure.
 */
export interface WebSocketMessage {
    type: MessageType;
    timestamp: number;
    payload: Record<string, unknown>;
    message_id?: string;
}

/**
 * Feedback delivery message from backend.
 */
export interface FeedbackDeliveryPayload {
    items: FeedbackItem[];
    request_id?: string;
    triggered_by: "reactive" | "proactive" | "manual";
    user_state_score?: number;
}

/**
 * System status update message.
 */
export interface SystemStatusMessage {
    status: SystemStatus;
    timestamp: number;
    eye_tracker_connected: boolean;
    vscode_connected: boolean;
    operation_mode: "reactive" | "proactive";
    eye_samples_processed: number;
    code_window_samples_processed: number;
    feedback_generated: number;
    experiment_active: boolean;
    llm_model?: string;
    experiment_id?: string;
    participant_id?: string;
    error_message?: string;
}

/**
 * Context request from backend.
 */
export interface ContextRequestPayload {
    request_id: string;
    include_file_content: boolean;
    include_diagnostics: boolean;
    include_visible_range: boolean;
    active_file_only: boolean;
}
