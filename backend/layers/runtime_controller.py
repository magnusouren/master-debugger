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
from datetime import datetime, timezone
import asyncio


from backend.api.serialization import json_safe
from backend.layers.signal_processing import SignalProcessingLayer
from backend.layers.forecasting_tool import ForecastingTool
from backend.layers.reactive_tool import ReactiveTool
from backend.layers.feedback_layer import FeedbackLayer
from backend.services.logger_service import get_logger
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

        # Generate session ID only if both participant_id and experiment_id are available
        if self._config.controller.participant_id and self._config.controller.experiment_id:
            self._session_id: Optional[str] = f"{self._config.controller.participant_id}_{self._config.controller.experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        else:
            self._session_id: Optional[str] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        
        # Initialize logger
        self._logger = get_logger()
        self._logger.system(
            "runtime_controller_initialized",
            {
                "operation_mode": self._operation_mode.name,
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id,
            },
            level="DEBUG",
        )
    
    async def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful.
        """
        self._logger.system("runtime_controller_initializing", {}, level="DEBUG")
        self._status = SystemStatus.READY
        self._stats["session_start"] = asyncio.get_event_loop().time()
        
        llm_ready = self._feedback_layer.initialize_llm()
        if not llm_ready:
            self._logger.system("llm_not_configured", {"fallback": "heuristics"}, level="WARNING")

        # Start main loop as background task - store it to prevent garbage collection
        main = asyncio.create_task(self._run_main_loop())
        
        self._logger.system("runtime_controller_ready",
                            {"status": self._status.name}, level="DEBUG")
        return True
    
    async def shutdown(self) -> None:
        """Shutdown all system components gracefully."""
        self._logger.system("runtime_controller_shutdown", {"final_stats": self._stats}, level="INFO")
        self._status = SystemStatus.DISCONNECTED

    
    def configure(self, config: SystemConfig) -> None:
        """
        Update system configuration.
        
        Args:
            config: New system configuration.
        """
        self._config = config
        self._operation_mode = config.controller.operation_mode
        
        # Reconfigure layers
        self._signal_processing.configure(config.signal_processing)
        self._forecasting.configure(config.forecasting)
        self._reactive_tool.configure(config.reactive_tool)
        self._feedback_layer.configure(config.feedback_layer)
        
        self._logger.system(
            "runtime_controller_reconfigured",
            {"operation_mode": self._operation_mode.name},
            level="INFO",
        )
    
    def set_operation_mode(self, mode: OperationMode) -> None:
        """
        Set the system operation mode.
        
        Args:
            mode: REACTIVE or PROACTIVE mode.
        """
        # Keep the stored configuration in sync with the current operation mode
        if self._config is not None and getattr(self._config, "controller", None) is not None:
            self._config.controller.operation_mode = mode
        
        self._logger.system(
            "operation_mode_changed",
            {"new_mode": self._operation_mode.name},
            level="INFO",
        )

        self._logger.experiment(
            "operation_mode_changed",
            {"new_mode": self._operation_mode.name},
            level="INFO",
        )

        # Reconfigure layers as needed using the updated configuration
        if self._config is not None:
            self.configure(self._config)
    
    def get_operation_mode(self) -> OperationMode:
        """
        Get current operation mode.
        
        Returns:
            Current operation mode.
        """
        return self._operation_mode
    
    def get_status(self) -> SystemStatus:
        """
        Get current system status.
        
        Returns:
            Current system status.
        """
        return self._status
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary of statistics.
        """
        return self._stats
    
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
        
        if self._current_code_context.metadata is None:
            self._current_code_context.metadata = {}

        self._current_code_context.metadata["experiment_id"] = self._experiment_id
        self._current_code_context.metadata["participant_id"] = self._participant_id    
        self._current_code_context.metadata["session_id"] = self._session_id    

        
        self._logger.experiment(
            "context_update_received",
            {
                "metadata": self._current_code_context.metadata,
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id
            },
            level="DEBUG",
        )

        self._logger.system(
            "context_update_processed",
            {
                "file": context.file_path, 
                "line": context.cursor_position.line,
                "char": context.cursor_position.character
            },
            level="INFO",
        )

        if self.should_generate_feedback():
            feedback = await self.trigger_feedback_generation()
            if feedback:
                # Send feedback back to VS Code
                await self.send_feedback(feedback, client_id=context.metadata.get("client_id"))

    async def send_feedback(self, feedback: FeedbackResponse, client_id: Optional[str] = None) -> bool:
        """
        Send feedback to VS Code for display.
        
        Args:
            feedback: Feedback to send.
            client_id: Optional target client ID.
            
        Returns:
            True if sent successfully.
        """
        
        try:
            msg = WebSocketMessage(
                type=MessageType.FEEDBACK_DELIVERY,
                timestamp=datetime.now(timezone.utc).timestamp(),
                payload=json_safe(feedback),
                message_id=None, # not a response to a specific message, so no message_id needed
                target_client_id=client_id
            )
            await self._emit(msg)
            self._logger.system(
                "feedback_sent",
                {"item_count": len(feedback.items), "client_id": client_id},
                level="INFO",
            )
            return True
        except Exception as e:
            self._logger.system(
                "feedback_send_error",
                {"error": str(e), "client_id": client_id},
                level="ERROR",
            )
            return False
    
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
        self._websocket_callbacks.append(callback)
        
    # --- Messaging ---

    async def _emit(self, message: WebSocketMessage) -> None:
        """
        Emit a message through registered websocket callbacks.
        
        Args:
            message: Message to emit.
        """
        for callback in self._websocket_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self._logger.system(
                    "websocket_callback_error",
                    {"error": str(e)},
                    level="ERROR",
                )
    
    # --- Feedback Control ---
    
    def should_generate_feedback(self) -> bool:
        """
        Determine if feedback should be generated based on current state.

        TODO - implement cooldowns, user state checks, etc.
        
        Returns:
            True if feedback should be generated.
        """
        if self._status != SystemStatus.READY:
            return False

        if self._current_code_context is None:
            return False

        return True 
    
    async def trigger_feedback_generation(self) -> Optional[FeedbackResponse]:
        """
        Trigger feedback generation if conditions are met.
        
        Returns:
            Generated feedback or None.
        """

        if self._current_code_context is None:
            self._logger.system(
                "feedback_generation_no_context",
                {},
                level="DEBUG",
            )
            return None


        feedback = await self._feedback_layer.generate_feedback_cached(
            context=self._current_code_context,
            user_state=self._current_user_state,
            feedback_types=None, # TODO - decide if different types needed
        )

        if feedback is not None:
            self._last_feedback_time = asyncio.get_event_loop().time()
            self._stats["feedback_generated"] = self._stats.get("feedback_generated",0) + 1
            return feedback
        else:
            self._logger.system(
                "feedback_generation_no_feedback",
                {},
                level="DEBUG",
            )
            return None
    

    def get_feedback_cooldown_remaining(self) -> float:
        """
        Get remaining cooldown time before next feedback.
        
        Returns:
            Remaining seconds in cooldown.
        """
        pass # TODO: Implement cooldown logic
    
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

        self._logger.system("runtime_controller_main_loop_started", {}, level="DEBUG")
        while self._status == SystemStatus.READY:
            await asyncio.sleep(0.1)  # Main loop tick
            
            # Update statistics
            self._update_statistics()
            
            # # Check for feedback generation
            # if self.should_generate_feedback():
            #     feedback = await self.trigger_feedback_generation()
            #     if feedback:
            #         # Send feedback back to VS Code
            #         await self.send_feedback(feedback)

        self._logger.system("runtime_controller_main_loop_ended", {}, level="DEBUG")
    
    def _update_statistics(self) -> None:
        """
        Update runtime statistics. 
        # TODO implement more stats as needed
        Currently tracks uptime.
        """
        if self._stats["session_start"] is not None:
            elapsed = asyncio.get_event_loop().time() - self._stats["session_start"]
            self._stats["uptime_seconds"] = elapsed