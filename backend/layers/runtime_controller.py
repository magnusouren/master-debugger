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
import contextlib
from pyexpat import features
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timezone
import asyncio


from backend.layers.signal_processing import SignalProcessingLayer
from backend.layers.forecasting_tool import ForecastingTool
from backend.layers.reactive_tool import ReactiveTool
from backend.layers.feedback_layer import FeedbackLayer
from backend.services.logger_service import get_logger
from backend.types.code_context import CodeContext
from backend.types.config import OperationMode, SystemConfig
from backend.types.eye_tracking import PredictedFeatures, WindowFeatures
from backend.types.feedback import FeedbackInteraction, FeedbackResponse
from backend.types.messages import SystemStatus, SystemStatusMessage
from backend.types.domain_events import DomainEvent, DomainEventType
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
            "eye_samples_processed": 0,
            "code_window_samples_processed": 0,
            "feedback_generated": 0,
            "session_start": None,
        }

        # eye tracker connection state
        self._eye_tracker_connected: bool = False
        
        # Domain event handlers for external communication
        self._event_handlers: List[Callable[[DomainEvent], None]] = []
        
        # Event loop reference
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Experiment tracking
        self._experiment_is_active: bool = False
        self._experiment_id: Optional[str] = self._config.controller.experiment_id
        self._participant_id: Optional[str] = self._config.controller.participant_id

        # Generate session ID only if both participant_id and experiment_id are available
        if self._config.controller.participant_id and self._config.controller.experiment_id:
            self._session_id: Optional[str] = f"{self._config.controller.participant_id}_{self._config.controller.experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        else:
            self._session_id: Optional[str] = None
        
        # main loop
        self._main_loop_task: Optional[asyncio.Task] = None

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

        if not self._experiment_id or not self._participant_id:
            self._logger.system(
                "runtime_controller_ready_no_experiment",
                {"warning": "Running without experiment context"},
                level="WARNING",
            )
        else:
            self._logger.system(
                "runtime_controller_ready_with_experiment",
                {
                    "experiment_id": self._experiment_id,
                    "participant_id": self._participant_id,
                    "session_id": self._session_id
                },
                level="INFO",
            )

        # Start main loop as background task - store it to prevent garbage collection
        self._main_loop_task = asyncio.create_task(self._run_main_loop())
        
        self._logger.system("runtime_controller_ready",
                            {"status": self._status.name}, level="DEBUG")
        return True
    
    async def shutdown(self) -> None:
        """Shutdown all system components gracefully."""
        self._logger.system("runtime_controller_shutdown", {"final_stats": self._stats}, level="INFO")
        self._status = SystemStatus.DISCONNECTED

        if self._main_loop_task:
            self._main_loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main_loop_task

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
        self._operation_mode = mode

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
    
    def get_system_status(self) -> SystemStatusMessage:
        """
        Get detailed system status message.
        
        Returns:
            SystemStatusMessage with current status details.
        """
        return SystemStatusMessage(
            status=self.get_status(),
            timestamp=datetime.now(timezone.utc).timestamp(),
            eye_tracker_connected=self.is_eye_tracker_connected(),
            operation_mode=self._operation_mode.name,
            eye_samples_processed=self._stats["eye_samples_processed"],
            code_window_samples_processed=self._stats["code_window_samples_processed"],
            feedback_generated=self._stats["feedback_generated"],
            llm_model=self._feedback_layer.get_llm_client().get_model_name() if self._feedback_layer.get_llm_client() else None,
            experiment_active=self._experiment_is_active,
            experiment_id=self._experiment_id,
            participant_id=self._participant_id,
        )
    
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
        return self._eye_tracker_connected
    
    # --- VS Code Communication ---
    
    async def handle_context_update(self, context: CodeContext) -> None:
        self._stats["code_window_samples_processed"] += 1

        incoming_meta = context.metadata or {}
        experiment_meta = {
            "experiment_id": self._experiment_id,
            "participant_id": self._participant_id,
            "session_id": self._session_id,
        }
        merged_meta = {**incoming_meta, **experiment_meta}

        context.metadata = merged_meta
        self._current_code_context = context

        self._logger.experiment(
            "context_update_received",
            {
                "metadata": merged_meta,
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id,
            },
            level="DEBUG",
        )

        self._logger.system(
            "context_update_processed",
            {
                "file": context.file_path,
                "line": context.cursor_position.line,
                "char": context.cursor_position.character,
            },
            level="INFO",
        )

        if self.should_generate_feedback():
            feedback = await self.trigger_feedback_generation()
            if feedback:
                recipient_id = merged_meta.get("requester_id")
                event_meta = {"recipient_id": recipient_id} if recipient_id else {}

                self._publish(DomainEvent(
                    event_type=DomainEventType.FEEDBACK_READY,
                    payload=feedback,
                    metadata=event_meta,
                ))

                self._logger.system(
                    "feedback_ready_published",
                    {"item_count": len(feedback.items), "recipient_id": recipient_id},
                    level="INFO",
                )   
    
    async def handle_feedback_interaction(
        self, interaction: FeedbackInteraction
    ) -> None:
        """
        Handle user interaction with feedback.
        
        Args:
            interaction: User interaction data.
        """
        pass  # TODO: Implement interaction handling
    
    def register_event_handler(
        self, handler: Callable[[DomainEvent], None]
    ) -> None:
        """
        Register a handler for domain events.
        
        Args:
            handler: Function to call with domain events.
        """
        self._event_handlers.append(handler)
        
    # --- Domain Event Publishing ---

    def _publish(self, event: DomainEvent) -> None:
        """
        Publish a domain event to all registered handlers.
        
        Args:
            event: Domain event to publish.
        """
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                self._logger.system(
                    "event_handler_error",
                    {"error": str(e), "event_type": event.event_type.value},
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
        self._stats["eye_samples_processed"] = self._stats.get("eye_samples_processed", 0) + 1

        if self._operation_mode == OperationMode.REACTIVE:
            # Baseline: observed features -> reactive
            self._reactive_tool.add_features(features)
            return

        # Proactive: observed features to forecasting
        self._forecasting.add_features(features)

    
    def _on_predicted_features(self, predicted: PredictedFeatures) -> None:
        """
        Handle predicted features from Forecasting Tool.
        
        Args:
            predicted: Predicted features.
        """

        # Nothing to do in reactive mode
        if self._operation_mode == OperationMode.REACTIVE:
            return
        
        # In proactive mode, pass predicted features to Reactive Tool
        self._reactive_tool.add_features(predicted)
    
    def _on_user_state_estimate(self, estimate: UserStateEstimate) -> None:
        """
        Handle user state estimate from Reactive Tool.
        
        Args:
            estimate: User state estimate.
        """
        self._current_user_state = estimate

        self._logger.system(
            "user_state_estimate_updated",
            {
                "score": estimate.user_state_score,
                "details": estimate.details,
            },
            level="DEBUG",
        )

        self._logger.experiment(
            "user_state_estimate_logged",
            {
                "score": estimate.user_state_score,
                "details": estimate.details,
            },
            level="DEBUG",
        )
    
    # --- Logging and Experiment Control ---
    
    async def start_experiment(
        self, 
        experiment_id: str, 
        participant_id: str
    ) -> SystemStatusMessage:
        """
        Start a new experiment session.
        
        Args:
            experiment_id: Unique experiment identifier.
            participant_id: Unique participant identifier.
        """

        self._experiment_id = experiment_id
        self._participant_id = participant_id
        self._experiment_is_active = True
        self._session_id = f"{participant_id}_{experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

        self._logger.system(
            "experiment_started",
            {
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id
            },
            level="INFO",
        )

        self._logger.experiment(
            "experiment_started",
            {
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id
            },
            level="INFO",
        )

        # Publish domain event for experiment start
        self._publish(DomainEvent(
            event_type=DomainEventType.EXPERIMENT_STARTED,
            payload=self.get_system_status(),
            metadata={
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id,
            },
        ))

        return self.get_system_status()

    def end_experiment(self) -> SystemStatusMessage:
        """End the current experiment session."""
        self._logger.system(
            "experiment_ended",
            {
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id
            },
            level="INFO",
        )

        self._logger.experiment(
            "experiment_ended",
            {
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id
            },
            level="INFO",
        )
        
        # Mark experiment as inactive before exporting data to avoid logging new entries during export
        self._experiment_is_active = False

        self.export_experiment_data()

        # Publish domain event for experiment end (before clearing IDs)
        self._publish(DomainEvent(
            event_type=DomainEventType.EXPERIMENT_ENDED,
            payload=self.get_system_status(),
            metadata={
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id,
            },
        ))

        self._experiment_id = None
        self._participant_id = None
        self._session_id = None

        return self.get_system_status()

    def get_experiment_data(self) -> Dict[str, Any]:
        """
        Get collected experiment data.
        
        Returns:
            Dictionary of experiment data.
        """
        if self._experiment_id is None or self._participant_id is None:
            self._logger.system(
                "get_experiment_data_no_experiment",
                {},
                level="WARNING",
            )
            return {}
        
        return self._logger.get_experiment_logs() # TODO - filter by experiment/participant/session if needed
    
    def export_experiment_data(self) -> bool:
        """
        Export experiment data to file.
            
        Returns:
            True if export successful.
        """
        if self._experiment_id is None or self._participant_id is None:
            self._logger.system(
                "export_experiment_data_no_experiment",
                {},
                level="WARNING",
            )
            return False


        filepath = f"logs/experiments/experiment_{self._session_id}.csv"
        success = self._logger.export_experiment_logs(filepath)
        return success


    # --- Internal Methods ---
    
    def _setup_layer_callbacks(self) -> None:
        """Set up callbacks between layers for data flow."""
        
        # Signal Processing calls back to Runtime Controller
        self._signal_processing.register_window_features_callback(
            self._on_window_features
        )

        # Forecasting Tool calls back to Runtime Controller
        self._forecasting.register_predicted_features_callback(
            self._on_predicted_features
        )

        # Reactive Tool calls back to Runtime Controller
        self._reactive_tool.register_user_state_callback(
            self._on_user_state_estimate
        )

        self._logger.system("layer_callbacks_configured", {}, level="DEBUG")

    
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
            await asyncio.sleep(1)  # Main loop tick
            
            # Update statistics
            self._update_statistics()

            # Publish system status periodically
            self._broadcast_system_status()
            
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

    def _broadcast_system_status(self) -> None:
        """
        Publish current system status as a domain event.
        """
        status_msg = self.get_system_status()
        self._publish(DomainEvent(
            event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
            payload=status_msg,
        ))