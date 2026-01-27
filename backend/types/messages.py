"""
Type definitions for WebSocket and API messages.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from .code_context import CodeContext
from .feedback import FeedbackItem, FeedbackInteraction
from .user_state import UserStateEstimate


class MessageType(Enum):
    """Types of WebSocket messages."""
    # From VS Code to Backend
    CONTEXT_UPDATE = "context_update"
    CONTEXT_REQUEST = "context_request"
    FEEDBACK_INTERACTION = "feedback_interaction"
    
    # From Backend to VS Code
    FEEDBACK_DELIVERY = "feedback_delivery"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    
    # Bidirectional
    PING = "ping"
    PONG = "pong"
    CONFIG_UPDATE = "config_update"


class SystemStatus(Enum):
    """System status states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class WebSocketMessage:
    """Base WebSocket message structure."""
    type: MessageType
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    message_id: Optional[str] = None
    target_client_id: Optional[str] = None  # For targeted messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "message_id": self.message_id,
            "target_client_id": self.target_client_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSocketMessage":
        """Create message from dictionary."""
        pass  # TODO: Implement


@dataclass
class ContextRequest:
    """Request for code context from backend to VS Code."""
    request_id: str
    timestamp: float
    
    # What context is needed
    include_file_content: bool = True
    include_diagnostics: bool = True
    include_visible_range: bool = True
    
    # Scope
    active_file_only: bool = True


@dataclass
class ContextUpdate:
    """Code context update from VS Code to backend."""
    request_id: Optional[str] = None  # If responding to a request
    context: CodeContext = field(default_factory=CodeContext)


@dataclass
class FeedbackMessage:
    """Feedback delivery message from backend to VS Code."""
    items: List[FeedbackItem] = field(default_factory=list)
    request_id: Optional[str] = None
    
    # Triggering information
    triggered_by: str = "reactive"  # "reactive", "proactive", "manual"
    user_state_score: Optional[float] = None


@dataclass
class SystemStatusMessage:
    """System status update message."""
    status: SystemStatus
    timestamp: float
    
    # Component statuses
    eye_tracker_connected: bool = False
    vscode_connected: bool = False
    
    # Current state
    operation_mode: str = "reactive"
    
    # Statistics
    samples_processed: int = 0
    feedback_generated: int = 0

    # Experiment status
    experiment_active: bool = False
    experiment_id: Optional[str] = None

    # Participant ID
    participant_id: Optional[str] = None
    
    # Error information
    error_message: Optional[str] = None
