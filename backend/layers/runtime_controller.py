"""
Runtime Controller

The Control Layer acts as a central orchestrator, maintaining global system 
state and coordinating data flow, mode selection, feedback timing, and 
communication between all backend components and the VS Code extension.

Responsibilities:
- Orchestrates all other layers and manages communication with the VS Code 
  extension and the eye-tracking interface
- Deciding when to request feedback based on user_state_score (reactive or 
  proactive), cooldowns, and interaction history
- Configuring layers for reactive and proactive modes
- Logging and experiment control
"""
from typing import Awaitable, Optional, Dict, Any, List, Callable
from datetime import datetime
from enum import Enum
import asyncio


from backend.layers.signal_processing import SignalProcessingLayer
from backend.layers.forecasting_tool import ForecastingTool
from backend.layers.reactive_tool import ReactiveTool
from backend.layers.feedback_layer import FeedbackLayer
from backend.types.code_context import CodeContext
from backend.types.config import OperationMode, SystemConfig
from backend.types.eye_tracking import PredictedFeatures, WindowFeatures
from backend.types.feedback import FeedbackInteraction, FeedbackResponse
from backend.types.messages import MessageType, SystemStatus, WebSocketMessage
from backend.types.user_state import UserStateEstimate


class RuntimeController:
    """
    Central orchestrator for the eye-tracking debugger system.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the Runtime Controller.
        
        Args:
            config: Complete system configuration.
        """
        self._config = config or SystemConfig()
        
        # Initialize layers
        self._signal_processing = SignalProcessingLayer(
            self._config.signal_processing
        )
        self._forecasting = ForecastingTool(self._config.forecasting)
        self._reactive_tool = ReactiveTool(self._config.reactive_tool)
        self._feedback_layer = FeedbackLayer(self._config.feedback_layer)
        
        # State
        self._status: SystemStatus = SystemStatus.INITIALIZING
        self._operation_mode: OperationMode = self._config.controller.operation_mode
        self._last_feedback_time: float = 0.0
        self._current_user_state: Optional[UserStateEstimate] = None
        self._current_code_context: Optional[CodeContext] = None
        
        # Statistics
        self._stats: Dict[str, Any] = {
            "samples_processed": 0,
            "feedback_generated": 0,
            "session_start": None,
        }
        
        # Callbacks for external communication
        self._websocket_callbacks: List[Callable[[WebSocketMessage], Awaitable[None]]] = []
        
        # Event loop reference
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Experiment tracking
        self._experiment_id: Optional[str] = self._config.controller.experiment_id
        self._participant_id: Optional[str] = self._config.controller.participant_id
    
    async def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful.
        """
        print("  Initializing runtime controller...")
        self._status = SystemStatus.READY
        self._stats["session_start"] = asyncio.get_event_loop().time()
        return True
    
    async def shutdown(self) -> None:
        """Shutdown all system components gracefully."""
        print("  Shutting down runtime controller...")
        self._status = SystemStatus.DISCONNECTED
    
    def configure(self, config: SystemConfig) -> None:
        """
        Update system configuration.
        
        Args:
            config: New system configuration.
        """
        pass  # TODO: Implement configuration update
    
    def set_operation_mode(self, mode: OperationMode) -> None:
        """
        Set the system operation mode.
        
        Args:
            mode: REACTIVE or PROACTIVE mode.
        """
        pass  # TODO: Implement mode switching
    
    def get_operation_mode(self) -> OperationMode:
        """
        Get current operation mode.
        
        Returns:
            Current operation mode.
        """
        pass  # TODO: Implement mode getter
    
    def get_status(self) -> SystemStatus:
        """
        Get current system status.
        
        Returns:
            Current system status.
        """
        pass  # TODO: Implement status getter
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary of statistics.
        """
        pass  # TODO: Implement statistics getter
    
    # --- Eye Tracker Interface ---
    
    async def connect_eye_tracker(self, device_id: Optional[str] = None) -> bool:
        """
        Connect to the Tobii eye tracker.
        
        Args:
            device_id: Optional specific device to connect to.
            
        Returns:
            True if connection successful.
        """
        pass  # TODO: Implement eye tracker connection
    
    async def disconnect_eye_tracker(self) -> None:
        """Disconnect from the eye tracker."""
        pass  # TODO: Implement eye tracker disconnection
    
    def is_eye_tracker_connected(self) -> bool:
        """
        Check if eye tracker is connected.
        
        Returns:
            True if connected.
        """
        pass  # TODO: Implement connection check
    
    # --- VS Code Communication ---
    
    async def handle_context_update(self, context: CodeContext) -> None:
        """
        Handle code context update from VS Code.
        First step in the data flow pipeline in the Control Layer.
        
        Args:
            context: Updated code context.
        """
        self._current_code_context = context
        self._stats["samples_processed"] += 1
        
        print(f"  [Runtime Controller] Context update file: {context.file_path}, "
              f"cursor: L{context.cursor_position.line if context.cursor_position else '?'}")
        
        # Check if feedback should be generated
        if self.should_generate_feedback():
            feedback = await self.trigger_feedback_generation()
            if feedback:
                print(f"    → Generated {len(feedback.items)} feedback items")
                self._stats["feedback_generated"] += 1
    
    async def request_context(self) -> None:
        msg = WebSocketMessage(
            type=MessageType.CONTEXT_REQUEST,
            timestamp=int(datetime.utcnow().timestamp() * 1000),
            payload={},
            message_id=None,
        )
        await self._emit(msg)
        
    async def send_feedback(self, feedback: FeedbackResponse) -> bool:
        msg = WebSocketMessage(
            type=MessageType.FEEDBACK_DELIVERY,
            timestamp=int(datetime.utcnow().timestamp() * 1000),
            payload=feedback.to_dict(),   # TODO - implement
            message_id=None,
        )
        await self._emit(msg)
        return True
    
    async def handle_feedback_interaction(
        self, interaction: FeedbackInteraction
    ) -> None:
        """
        Handle user interaction with feedback.
        
        Args:
            interaction: User interaction data.
        """
        pass  # TODO: Implement interaction handling
    
    def register_websocket_callback(
        self, callback: Callable[[WebSocketMessage], None]
    ) -> None:
        """
        Register callback for WebSocket messages.
        
        Args:
            callback: Function to call with messages.
        """
        pass  # TODO: Implement callback registration
    
    # --- Feedback Control ---
    
    def should_generate_feedback(self) -> bool:
        """
        Determine if feedback should be generated based on current state.
        
        Returns:
            True if feedback should be generated.
        """
        pass  # TODO: Implement feedback decision logic
    
    async def trigger_feedback_generation(self) -> Optional[FeedbackResponse]:
        """
        Trigger feedback generation if conditions are met.
        
        Returns:
            Generated feedback or None.
        """
        pass  # TODO: Implement feedback triggering
    
    def get_feedback_cooldown_remaining(self) -> float:
        """
        Get remaining cooldown time before next feedback.
        
        Returns:
            Remaining seconds in cooldown.
        """
        pass  # TODO: Implement cooldown calculation
    
    # --- Data Flow Callbacks ---
    
    def _on_window_features(self, features: WindowFeatures) -> None:
        """
        Handle new window features from Signal Processing.
        
        Args:
            features: Computed window features.
        """
        pass  # TODO: Implement feature handling
    
    def _on_predicted_features(self, predicted: PredictedFeatures) -> None:
        """
        Handle predicted features from Forecasting Tool.
        
        Args:
            predicted: Predicted features.
        """
        pass  # TODO: Implement prediction handling
    
    def _on_user_state_estimate(self, estimate: UserStateEstimate) -> None:
        """
        Handle user state estimate from Reactive Tool.
        
        Args:
            estimate: User state estimate.
        """
        pass  # TODO: Implement state estimate handling
    
    # --- Logging and Experiment Control ---
    
    def start_experiment(
        self, 
        experiment_id: str, 
        participant_id: str
    ) -> None:
        """
        Start a new experiment session.
        
        Args:
            experiment_id: Unique experiment identifier.
            participant_id: Unique participant identifier.
        """
        pass  # TODO: Implement experiment start
    
    def end_experiment(self) -> None:
        """End the current experiment session."""
        pass  # TODO: Implement experiment end
    
    def log_event(
        self, 
        event_type: str, 
        data: Dict[str, Any]
    ) -> None:
        """
        Log an event for the experiment.
        
        Args:
            event_type: Type of event.
            data: Event data.
        """
        pass  # TODO: Implement event logging
    
    def get_experiment_data(self) -> Dict[str, Any]:
        """
        Get collected experiment data.
        
        Returns:
            Dictionary of experiment data.
        """
        pass  # TODO: Implement data retrieval
    
    def export_experiment_data(self, path: str) -> bool:
        """
        Export experiment data to file.
        
        Args:
            path: Path to export file.
            
        Returns:
            True if export successful.
        """
        pass  # TODO: Implement data export
    
    # --- Internal Methods ---
    
    def _setup_layer_callbacks(self) -> None:
        """Set up callbacks between layers for data flow."""
        pass  # TODO: Implement callback setup
    
    def _validate_system_state(self) -> bool:
        """
        Validate that system is in a valid state.
        
        Returns:
            True if state is valid.
        """
        pass  # TODO: Implement state validation
    
    async def _run_main_loop(self) -> None:
        """Main processing loop."""
        pass  # TODO: Implement main loop
    
    def _update_statistics(self) -> None:
        """Update internal statistics."""
        pass  # TODO: Implement statistics update

    async def _emit(self, msg: WebSocketMessage) -> None:
        """
        Emit a WebSocket message via registered callbacks.
        
        :param self: Description
        :param msg: WebSocket message to emit.
        :type msg: WebSocketMessage 
        """
        for cb in self._websocket_callbacks:
            await cb(msg)