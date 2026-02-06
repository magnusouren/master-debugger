import { useState } from "react";
import type { FeedbackItem } from "../types";

interface FeedbackListProps {
  items: FeedbackItem[];
  onInteraction: (feedbackId: string, interactionType: "dismissed" | "accepted") => void;
}

export function FeedbackList({ items, onInteraction }: FeedbackListProps) {
  if (items.length === 0) {
    return (
      <div className="empty-state">
        No feedback available. Start coding and feedback will appear here based on your activity.
      </div>
    );
  }

  return (
    <div className="feedback-list">
      {items.map((item) => (
        <FeedbackAlertCard
          key={item.metadata.feedback_id}
          item={item}
          onAccept={() => onInteraction(item.metadata.feedback_id, "accepted")}
          onDismiss={() => onInteraction(item.metadata.feedback_id, "dismissed")}
        />
      ))}
    </div>
  );
}

interface FeedbackAlertCardProps {
  item: FeedbackItem;
  onDismiss: () => void;
  onAccept: () => void;
}

function FeedbackAlertCard({ item, onDismiss, onAccept }: FeedbackAlertCardProps) {
  const [accepted, setAccepted] = useState(false);

  const handleAccept = () => {
    setAccepted(true);
    onAccept();
  };

  const handleDismiss = () => {
    onDismiss();
  };

  return (
    <>
      {!accepted ? (
        <div className={`feedback-item ${item.feedback_type}`}>
          <div className="feedback-header">
            <span className="feedback-title">Feedback Available</span>
            {/* <span className="feedback-priority">{item.priority}</span> */}
          </div>
          <p className="feedback-message">Do you want to be presented this feedback?</p>
          <div className="feedback-actions">
            {item.actionable && (
              <button className="feedback-action-btn" onClick={handleAccept} disabled={accepted}>
                Yes
              </button>
            )}
            {item.dismissible && (
              <button className="feedback-action-btn" onClick={handleDismiss}>
                No
              </button>
            )}
          </div>
        </div>
      ) : <FeedbackItemCard item={item} onDismiss={onDismiss} onAccept={() => { }} />}
    </>
  );
}

interface FeedbackItemCardProps {
  item: FeedbackItem;
  onDismiss: () => void;
  onAccept: () => void;
}

function FeedbackItemCard({ item, onDismiss, onAccept }: FeedbackItemCardProps) {
  return (
    <div className={`feedback-item ${item.feedback_type}`}>
      <div className="feedback-header">
        <span className="feedback-title">{item.title}</span>
        {/* <span className="feedback-priority">{item.priority}</span> */}
      </div>
      <p className="feedback-message">{item.message}</p>
      <div className="feedback-actions">
        {item.actionable && (
          <button className="feedback-action-btn" onClick={onAccept}>
            {item.action_label || "Apply"}
          </button>
        )}
        {item.dismissible && (
          <button className="feedback-action-btn" onClick={onDismiss}>
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
