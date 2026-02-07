import { useState, useEffect } from "react";
import type { FeedbackItem, InteractionType } from "../types";

interface FeedbackListProps {
  items: FeedbackItem[];
  onInteraction: (feedbackId: string, interactionType: InteractionType) => void;
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
          onInteraction={(type) => onInteraction(item.metadata.feedback_id, type)}
        />
      ))}
    </div>
  );
}

interface FeedbackAlertCardProps {
  item: FeedbackItem;
  onInteraction: (type: InteractionType) => void;
}

function FeedbackAlertCard({ item, onInteraction }: FeedbackAlertCardProps) {
  const [accepted, setAccepted] = useState(false);
  const [presented, setPresented] = useState(false);

  // Log when feedback is first presented to user
  useEffect(() => {
    if (!presented) {
      setPresented(true);
      onInteraction("presented");
    }
  }, [presented, onInteraction]);

  const handleAccept = () => {
    setAccepted(true);
    onInteraction("accepted");
  };

  const handleReject = () => {
    onInteraction("rejected");
  };

  const handleHighlight = () => {
    onInteraction("highlighted");
  };

  const handleDismiss = () => {
    onInteraction("dismissed");
  };

  return (
    <>
      {!accepted ? (
        <div className={`feedback-item ${item.feedback_type}`}>
          <div className="feedback-header">
            <span className="feedback-title">Feedback Available</span>
          </div>
          <p className="feedback-message">Do you want to be presented this feedback?</p>
          <div className="feedback-actions">
            {item.actionable && (
              <button className="feedback-action-btn" onClick={handleAccept} disabled={accepted}>
                Yes
              </button>
            )}
            {item.dismissible && (
              <button className="feedback-action-btn" onClick={handleReject}>
                No
              </button>
            )}
          </div>
        </div>
      ) : (
        <FeedbackItemCard
          item={item}
          onHighlight={handleHighlight}
          onDismiss={handleDismiss}
        />
      )}
    </>
  );
}

interface FeedbackItemCardProps {
  item: FeedbackItem;
  onHighlight: () => void;
  onDismiss: () => void;
}

function FeedbackItemCard({ item, onHighlight, onDismiss }: FeedbackItemCardProps) {
  return (
    <div className={`feedback-item ${item.feedback_type}`}>
      <div className="feedback-header">
        <span className="feedback-title">{item.title}</span>
      </div>
      <p className="feedback-message">{item.message}</p>
      <div className="feedback-actions">
        {item.code_range && (
          <button className="feedback-action-btn" onClick={onHighlight}>
            {item.action_label || "Highlight in Code"}
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
