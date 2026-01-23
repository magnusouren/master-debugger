# Type definitions for the eye-tracking debugger backend
from .eye_tracking import (
    RawGazeData,
    GazeSample,
    WindowFeatures,
    PredictedFeatures,
)
from .user_state import (
    UserStateScore,
    UserStateEstimate,
)
from .code_context import (
    CodePosition,
    CodeRange,
    DiagnosticInfo,
    CodeContext,
)
from .feedback import (
    FeedbackItem,
    FeedbackMetadata,
    FeedbackResponse,
    FeedbackInteraction,
    FeedbackType,
    FeedbackPriority,
)
from .config import (
    SignalProcessingConfig,
    ForecastingConfig,
    ReactiveToolConfig,
    ControllerConfig,
    SystemConfig,
    OperationMode,
)
from .messages import (
    MessageType,
    SystemStatus,
    WebSocketMessage,
    ContextRequest,
    ContextUpdate,
    FeedbackMessage,
    SystemStatusMessage,
)

__all__ = [
    # Eye tracking types
    "RawGazeData",
    "GazeSample",
    "WindowFeatures",
    "PredictedFeatures",
    # User state types
    "UserStateScore",
    "UserStateEstimate",
    # Code context types
    "CodePosition",
    "CodeRange",
    "DiagnosticInfo",
    "CodeContext",
    # Feedback types
    "FeedbackItem",
    "FeedbackMetadata",
    "FeedbackResponse",
    "FeedbackInteraction",
    "FeedbackType",
    "FeedbackPriority",
    # Config types
    "SignalProcessingConfig",
    "ForecastingConfig",
    "ReactiveToolConfig",
    "ControllerConfig",
    "SystemConfig",
    "OperationMode",
    # Message types
    "MessageType",
    "SystemStatus",
    "WebSocketMessage",
    "ContextRequest",
    "ContextUpdate",
    "FeedbackMessage",
    "SystemStatusMessage",
]