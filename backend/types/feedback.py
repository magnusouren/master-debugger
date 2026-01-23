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


@dataclass
class FeedbackInteraction:
    """
    User interaction with feedback (for logging).
    Received from VS Code extension.
    """
    feedback_id: str
    interaction_type: str  # "dismissed", "accepted", "clicked", "hovered"
    timestamp: float
    duration_ms: Optional[float] = None  # How long feedback was visible
    metadata: Dict[str, Any] = field(default_factory=dict)
