/**
 * Types shared between webview and extension
 * These mirror the extension types for use in React
 */

export enum FeedbackType {
  HINT = "hint",
  SUGGESTION = "suggestion",
  WARNING = "warning",
  EXPLANATION = "explanation",
  SIMPLIFICATION = "simplification",
}

export enum FeedbackPriority {
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
  CRITICAL = "critical",
}

export interface CodePosition {
  line: number;
  character: number;
}

export interface CodeRange {
  start: CodePosition;
  end: CodePosition;
  content?: string;
}

export interface FeedbackMetadata {
  generated_at: number;
  generation_time_ms: number;
  cached: boolean;
  feedback_id: string;
  model_used?: string;
  cache_key?: string;
  session_id?: string;
  extra?: Record<string, unknown>;
}

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
 * Types of user interactions with feedback.
 *
 * Flow:
 * 1. Feedback presented -> user accepts OR rejects
 * 2. If accepted -> user can highlight in code OR dismiss
 */
export type InteractionType =
  // Stage 1: Initial presentation
  | "presented" // Feedback was shown to user
  | "accepted" // User accepted to see feedback details
  | "rejected" // User rejected seeing the feedback
  // Stage 2: After accepting
  | "highlighted" // User clicked to highlight in code
  | "dismissed"; // User dismissed the shown feedback

export interface SystemStatus {
  status: "initializing" | "ready" | "processing" | "paused" | "error" | "disconnected";
  timestamp: number;
  eye_tracker_connected: boolean;
  operation_mode: "reactive" | "proactive";
  eye_samples_processed: number;
  code_window_samples_processed: number;
  feedback_generated: number;
  experiment_active: boolean;
  feedback_cooldown_left_s: number;
  llm_model?: string;
  experiment_id?: string;
  participant_id?: string;
  user_state_score?: number | null;
  error_message?: string;
}
