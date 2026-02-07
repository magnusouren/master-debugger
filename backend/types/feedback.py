"""
Type definitions for feedback generation and delivery.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from .code_context import CodeRange


class FeedbackType(Enum):
    """Types of feedback that can be generated."""
    HINT = "hint"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    EXPLANATION = "explanation"
    SIMPLIFICATION = "simplification"


class FeedbackPriority(Enum):
    """Priority levels for feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeedbackMetadata:
    """Metadata for a feedback item."""
    generated_at: float = 0.0  # Timestamp
    generation_time_ms: float = 0.0  # How long it took to generate
    model_used: Optional[str] = None  # LLM or method used
    cached: bool = False
    cache_key: Optional[str] = None
    
    # Tracking
    feedback_id: str = ""
    session_id: Optional[str] = None
    
    # Additional data
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackItem:
    """
    A single feedback item ready for rendering in VS Code.
    Output of Feedback Layer.
    """
    # Display content
    title: str  # Short title
    message: str  # Descriptive message
    
    # Type and priority
    feedback_type: FeedbackType = FeedbackType.HINT
    priority: FeedbackPriority = FeedbackPriority.MEDIUM
    
    # Code location (for highlighting in IDE)
    code_range: Optional[CodeRange] = None
    
    # Confidence
    confidence: float = 0.0  # 0.0 to 1.0
    
    # Actions available to user
    dismissible: bool = True
    actionable: bool = False
    action_label: Optional[str] = None
    
    # Metadata
    metadata: FeedbackMetadata = field(default_factory=FeedbackMetadata)


@dataclass
class FeedbackResponse:
    """
    Response containing one or more feedback items.
    Sent to VS Code extension.
    """
    items: List[FeedbackItem] = field(default_factory=list)
    
    # Request tracking
    request_id: Optional[str] = None
    
    # Timing
    total_generation_time_ms: float = 0.0
    
    # Status
    success: bool = True
    error_message: Optional[str] = None

    # Additional data
    metadata: FeedbackMetadata = field(default_factory=FeedbackMetadata)


class InteractionType(Enum):
    """
    Types of user interactions with feedback.
    
    Flow:
    1. Feedback presented -> user accepts OR rejects
    2. If accepted -> user can highlight in code OR dismiss
    """
    # Stage 1: Initial presentation
    PRESENTED = "presented"  # Feedback was shown to user
    ACCEPTED = "accepted"    # User accepted to see feedback details
    REJECTED = "rejected"    # User rejected seeing the feedback
    
    # Stage 2: After accepting
    HIGHLIGHTED = "highlighted"  # User clicked to highlight in code
    DISMISSED = "dismissed"      # User dismissed the shown feedback


@dataclass
class FeedbackInteraction:
    """
    User interaction with feedback (for logging).
    Received from VS Code extension.
    """
    feedback_id: str
    interaction_type: str  # See InteractionType enum
    timestamp: float  # When interaction occurred
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'FeedbackInteraction':
        return FeedbackInteraction(
            feedback_id=data.get("feedback_id", ""),
            interaction_type=data.get("interaction_type", ""),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )
