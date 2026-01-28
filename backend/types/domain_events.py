"""
Domain-level event types for the RuntimeController.

These events are transport-agnostic and represent domain state changes
that external systems (e.g., WebSocket adapters) can subscribe to.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class DomainEventType(Enum):
    """Types of domain events emitted by the RuntimeController."""
    
    # Feedback events
    FEEDBACK_READY = "feedback_ready"
    
    # System status events
    SYSTEM_STATUS_UPDATED = "system_status_updated"
    
    # Experiment lifecycle events
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_ENDED = "experiment_ended"


@dataclass
class DomainEvent:
    """
    A domain-level event emitted by the RuntimeController.
    
    Attributes:
        event_type: The type of domain event.
        timestamp: Unix timestamp when the event was created.
        payload: Event-specific domain object (e.g., FeedbackResponse, SystemStatusMessage).
        metadata: Optional metadata about the event context.
    """
    event_type: DomainEventType
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    payload: Any = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "metadata": self.metadata,
        }
