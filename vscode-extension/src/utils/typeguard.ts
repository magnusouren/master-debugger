import { StatusUpdatePayload } from "../types";

export function isStatusUpdatePayload(x: unknown): x is StatusUpdatePayload {
  if (!x || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;

  // minimum
  if (typeof o.status !== "string") return false;
  if (typeof o.eye_tracker_connected !== "boolean") return false;
  if (typeof o.vscode_connected !== "boolean") return false;
  if (typeof o.operation_mode !== "string") return false;
  if (typeof o.samples_processed !== "number") return false;
  if (typeof o.feedback_generated !== "number") return false;
  if (typeof o.experiment_active !== "boolean") return false;

  // optional
  if ("experiment_id" in o && o.experiment_id !== undefined && o.experiment_id !== null && typeof o.experiment_id !== "string") return false;
  if ("participant_id" in o && o.participant_id !== undefined && o.participant_id !== null && typeof o.participant_id !== "string") return false;
  if ("error_message" in o && o.error_message !== undefined && o.error_message !== null && typeof o.error_message !== "string") return false;

  return true;
}