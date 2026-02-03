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
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timezone
import asyncio


from backend.layers.signal_processing import SignalProcessingLayer
from backend.layers.forecasting_tool import ForecastingTool
from backend.layers.reactive_tool import ReactiveTool
from backend.layers.feedback_layer import FeedbackLayer
from backend.services.logger_service import get_logger
from backend.services.eye_tracker.factory import create_eye_tracker_adapter
from backend.services.eye_tracker.base import EyeTrackerAdapter
from backend.types.code_context import CodeContext
from backend.types.config import OperationMode, SystemConfig
from backend.types.eye_tracking import PredictedFeatures, WindowFeatures, GazeSample
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
        
        # Initialize logger first (shared by all layers)
        self._logger = get_logger()
        
        # Initialize layers with shared logger
        self._signal_processing = SignalProcessingLayer(
            self._config.signal_processing,
            logger=self._logger
        )
        self._forecasting = ForecastingTool(
            self._config.forecasting,
            logger=self._logger
        )
        self._reactive_tool = ReactiveTool(
            self._config.reactive_tool,
            logger=self._logger
        )
        self._feedback_layer = FeedbackLayer(
            self._config.feedback_layer,
            logger=self._logger
        )
        
        # State
        self._status: SystemStatus = SystemStatus.INITIALIZING
        self._operation_mode: OperationMode = self._config.controller.operation_mode
        self._last_feedback_time: float = 0.0
        self._current_user_state: Optional[UserStateEstimate] = None
        self._current_code_context: Optional[CodeContext] = None
        
        # Feedback pipeline state
        self._context_version: int = 0
        self._pending_feedback: Optional[FeedbackResponse] = None
        self._pending_feedback_version: int = 0
        self._last_delivered_version: int = 0
        self._feedback_generation_task: Optional[asyncio.Task] = None
        
        # Statistics TODO - figure out what to track
        self._stats: Dict[str, Any] = {
            "eye_samples_processed": 0,
            "code_window_samples_received": 0,
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
        
        # Eye tracker adapter
        self._eye_tracker_adapter: Optional[EyeTrackerAdapter] = None
        
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

        # Log initialization complete
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
        
        # Get event loop reference
        self._loop = asyncio.get_event_loop()
        
        # Initialize eye tracker adapter
        self._eye_tracker_adapter = create_eye_tracker_adapter(self._config, loop=self._loop)
        
        llm_ready = self._feedback_layer.initialize_llm()
        if not llm_ready:
            self._logger.system("llm_not_configured", {"fallback": "heuristics"}, level="WARNING")

        # Set up callbacks between layers
        self._setup_layer_callbacks()

        # Start main loop as background task - store it to prevent garbage collection
        self._main_loop_task = asyncio.create_task(self._run_main_loop())
        
        self._logger.system("runtime_controller_ready", self.get_system_status(), level="INFO")

        return True
    
    async def shutdown(self) -> None:
        """Shutdown all system components gracefully."""
        self._logger.system("runtime_controller_shutdown", {"final_stats": self._stats}, level="INFO")
        self._status = SystemStatus.DISCONNECTED

        # Disconnect eye tracker
        if self._eye_tracker_adapter:
            with contextlib.suppress(Exception):
                await self.disconnect_eye_tracker()

        # Cancel feedback generation task if running
        if self._feedback_generation_task and not self._feedback_generation_task.done():
            self._feedback_generation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._feedback_generation_task

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
            status=self.get_status().value,
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
        Connect to the eye tracker.
        
        Args:
            device_id: Optional specific device to connect to.
                      If None, uses config default or auto-selects first device.
            
        Returns:
            True if connection successful.
        """
        if not self._eye_tracker_adapter:
            self._logger.system(
                "eye_tracker_adapter_not_initialized",
                {},
                level="ERROR"
            )
            return False
        
        # Use config default if no device_id provided
        if device_id is None:
            device_id = self._config.eye_tracker.device_id
        
        try:
            # Attempt connection
            ok = await self._eye_tracker_adapter.connect(device_id=device_id)
            self._eye_tracker_connected = ok
            
            if ok:
                # Start signal processing
                self._signal_processing.start()
                
                # Start streaming
                await self._eye_tracker_adapter.start_streaming()
                
                device_info = self._eye_tracker_adapter.get_device_info()
                self._logger.system(
                    "eye_tracker_connected",
                    device_info,
                    level="INFO"
                )
                
                # Publish status update
                self._publish(DomainEvent(
                    event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
                    payload=self.get_system_status(),
                ))
            else:
                self._logger.system(
                    "eye_tracker_connection_failed",
                    {"device_id": device_id},
                    level="WARNING"
                )
                
                # Publish status update
                self._publish(DomainEvent(
                    event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
                    payload=self.get_system_status(),
                ))
            
            return ok
            
        except Exception as e:
            self._logger.system(
                "eye_tracker_connection_error",
                {"error": str(e), "device_id": device_id},
                level="ERROR"
            )
            self._eye_tracker_connected = False
            
            # Publish status update
            self._publish(DomainEvent(
                event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
                payload=self.get_system_status(),
            ))
            
            return False
    
    async def disconnect_eye_tracker(self) -> None:
        """Disconnect from the eye tracker."""
        if not self._eye_tracker_adapter:
            return
        
        try:
            # Stop streaming
            await self._eye_tracker_adapter.stop_streaming()
            
            # Disconnect
            await self._eye_tracker_adapter.disconnect()
            
            # Stop signal processing
            self._signal_processing.stop()
            
            self._eye_tracker_connected = False
            
            self._logger.system(
                "eye_tracker_disconnected",
                {},
                level="INFO"
            )
            
            # Publish status update
            self._publish(DomainEvent(
                event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
                payload=self.get_system_status(),
            ))
            
        except Exception as e:
            self._logger.system(
                "eye_tracker_disconnection_error",
                {"error": str(e)},
                level="ERROR"
            )
            self._eye_tracker_connected = False
    
    def is_eye_tracker_connected(self) -> bool:
        """
        Check if eye tracker is connected.
        
        Returns:
            True if connected.
        """
        return self._eye_tracker_connected
    
    # --- VS Code Communication ---
    
    async def handle_context_update(self, context: CodeContext) -> None:
        self._stats["code_window_samples_received"] += 1

        incoming_meta = context.metadata or {}
        experiment_meta = {
            "experiment_id": self._experiment_id,
            "participant_id": self._participant_id,
            "session_id": self._session_id,
        }
        merged_meta = {**incoming_meta, **experiment_meta}

        context.metadata = merged_meta
        self._current_code_context = context
        
        # Increment context version to track this update
        self._context_version += 1
        current_version = self._context_version

        self._logger.experiment(
            "context_update_received",
            {
                "metadata": merged_meta,
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id,
                "context_version": current_version,
            },
            level="DEBUG",
        )

        self._logger.system(
            "context_update_received",
            {
                "file": context.file_path,
                "line": context.cursor_position.line,
                "char": context.cursor_position.character,
                "context_version": current_version,
            },
            level="INFO",
        )

        # Cancel any ongoing feedback generation task (new context invalidates old)
        if self._feedback_generation_task is not None and not self._feedback_generation_task.done():
            self._feedback_generation_task.cancel()
            self._logger.system(
                "feedback_generation_cancelled",
                {"reason": "new_context_arrived", "cancelled_version": current_version - 1},
                level="DEBUG",
            )

        # Start async feedback generation for this context version
        if self._can_start_feedback_generation():
            self._feedback_generation_task = asyncio.create_task(
                self._generate_feedback_for_version(current_version, context)
            )   

    def manual_send_feedback(self) -> bool:
        """
        Manual trigger to deliver the latest available feedback immediately.

        Returns:
            True if something was delivered, False otherwise.
        """
        return self._try_deliver_feedback(force=True)
    
    async def handle_feedback_interaction(
        self, interaction: FeedbackInteraction
    ) -> bool:
        """
        Handle user interaction with feedback.
        
        Args:
            interaction: User interaction data.
        """

        category_msg = ""
        if interaction.interaction_type == "dismissed":
            category_msg = "feedback_dismissed_by_user"
        elif interaction.interaction_type == "accepted":
            category_msg = "feedback_accepted_by_user"
        else:
            category_msg = f"feedback_interaction_unknown_type: {interaction.interaction_type}"
        
        self._logger.system(
            category_msg,
            {
                "feedback_id": interaction.feedback_id,
                "action_taken": interaction.interaction_type,
                "timestamp": interaction.timestamp,
            },
            level="INFO",
        )
        self._logger.experiment(
            category_msg,
            {
                "feedback_id": interaction.feedback_id,
                "action_taken": interaction.interaction_type,
                "timestamp": interaction.timestamp,
            },
            level="INFO",
        )

        return True
    
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
    
    def _can_start_feedback_generation(self) -> bool:
        """
        Determine if feedback generation can be started.
        
        This checks preconditions for starting a new feedback generation task.
        Actual delivery decisions (cooldown, threshold) are checked at delivery time.
        
        Returns:
            True if feedback generation can be started.
        """
        if self._status != SystemStatus.READY:
            return False

        if self._current_code_context is None:
            return False

        return True
    
    def _should_deliver_feedback(self) -> bool:
        """
        Determine if feedback should be delivered based on current state.
        
        Checks:
        - System is ready
        - Pending feedback exists and hasn't been delivered yet
        - User state threshold is met (if configured)
        - Cooldown period has elapsed
        
        Returns:
            True if feedback should be delivered.
        """
        if self._status != SystemStatus.READY:
            return False

        # No pending feedback or already delivered
        if self._pending_feedback is None:
            self._logger.system(
                "no_pending_feedback_to_deliver",
                {},
                level="DEBUG",
            )
            return False
        
        if self._pending_feedback_version <= self._last_delivered_version:
            self._logger.system(
                "feedback_already_delivered",
                {
                    "pending_version": self._pending_feedback_version,
                    "last_delivered_version": self._last_delivered_version,
                },
                level="DEBUG",
            )
            return False

      
        if self.get_feedback_cooldown_remaining() > 0.0:
            self._logger.system(
                "feedback_delivery_cooldown",
                {
                    "remaining": self.get_feedback_cooldown_remaining(),
                    "pending_version": self._pending_feedback_version,
                },
                level="DEBUG",
            )
            return False

        # Check user state threshold (if user state is available)
        if self._current_user_state is not None:
            threshold = self._config.controller.min_score_for_feedback
            if self._current_user_state.score.score < threshold:
                self._logger.system(
                    "feedback_delivery_threshold_not_met",
                    {
                        "score": self._current_user_state.score.score,
                        "threshold": threshold,
                        "pending_version": self._pending_feedback_version,
                    },
                    level="DEBUG",
                )
                return False
            self._logger.system(
                "feedback_delivery_threshold_met",
                {
                    "score": self._current_user_state.score.score,
                    "threshold": threshold,
                    "pending_version": self._pending_feedback_version,
                },
                level="DEBUG",
            )
        return True
    
    def _try_deliver_feedback(self, force: bool = False) -> bool:
        """
        Attempt to deliver the latest pending feedback.

        If force=True, bypass threshold/cooldown checks and deliver the latest
        pending feedback if available.
        """
        if force:
            if self._status != SystemStatus.READY:
                return False
            if self._pending_feedback is None:
                return False

            feedback = self._pending_feedback
            version = self._pending_feedback_version

            # Optional: if you still want to prevent re-sending same version manually,
            # keep this check. If manual should re-send, remove it.
            # if version <= self._last_delivered_version:
            #     return False

        else:
            if not self._should_deliver_feedback():
                return False
            feedback = self._pending_feedback
            version = self._pending_feedback_version

        # Mark as delivered before publishing (prevents duplicates)
        self._last_delivered_version = max(self._last_delivered_version, version)
        self._last_feedback_time = asyncio.get_event_loop().time()

        recipient_id = None
        if self._current_code_context and self._current_code_context.metadata:
            recipient_id = self._current_code_context.metadata.get("requester_id")

        event_meta = {"recipient_id": recipient_id} if recipient_id else {}
        event_meta["feedback_version"] = version
        event_meta["trigger"] = "manual" if force else "user_state"

        self._publish(DomainEvent(
            event_type=DomainEventType.FEEDBACK_READY,
            payload=feedback,
            metadata=event_meta,
        ))

        self._logger.system(
            "feedback_delivered",
            {
                "item_count": len(feedback.items),
                "recipient_id": recipient_id,
                "feedback_version": version,
                "trigger": event_meta["trigger"],
            },
            level="INFO",
        )

        return True
    
    async def _generate_feedback_for_version(
        self, 
        version: int, 
        context: CodeContext
    ) -> None:
        """
        Generate feedback for a specific context version.
        
        This runs as a background task. The result is only stored if the
        version still matches the latest context version (i.e., no newer
        context has arrived during generation).
        
        Args:
            version: The context version this generation is for.
            context: The code context to generate feedback for.
        """
        try:
            self._logger.system(
                "feedback_generation_started",
                {"version": version},
                level="DEBUG",
            )

            feedback = await self._feedback_layer.generate_feedback_cached(
                context=context,
                user_state=self._current_user_state,
                feedback_types=None,
            )

            # Check if this version is still current
            if version != self._context_version:
                self._logger.system(
                    "feedback_generation_stale",
                    {
                        "generated_version": version,
                        "current_version": self._context_version,
                    },
                    level="DEBUG",
                )
                return

            if feedback is not None:
                self._pending_feedback = feedback
                self._pending_feedback_version = version
                self._stats["feedback_generated"] = self._stats.get("feedback_generated", 0) + 1

                self._logger.system(
                    "feedback_generation_completed",
                    {
                        "version": version,
                        "item_count": len(feedback.items),
                    },
                    level="INFO",
                )

                self._logger.experiment(
                    "feedback_generated",
                    {
                        "version": version,
                        "item_count": len(feedback.items),
                    },
                    level="INFO",
                )
            else:
                self._logger.system(
                    "feedback_generation_empty",
                    {"version": version},
                    level="DEBUG",
                )

        except asyncio.CancelledError:
            self._logger.system(
                "feedback_generation_cancelled",
                {"version": version},
                level="DEBUG",
            )
            raise
        except Exception as e:
            self._logger.system(
                "feedback_generation_error",
                {"version": version, "error": str(e)},
                level="ERROR",
            )
        

    def get_feedback_cooldown_remaining(self) -> float:
        """
        Get remaining cooldown time before next feedback.
        
        Returns:
            Remaining seconds in cooldown.
        """
        current_time = asyncio.get_event_loop().time()
        cooldown = self._config.controller.feedback_cooldown_seconds
        elapsed = current_time - self._last_feedback_time
        remaining = max(0.0, cooldown - elapsed)
        return remaining
    
    # --- Data Flow Callbacks ---
    
    def _on_gaze_samples(self, samples: List[GazeSample]) -> None:
        """
        Handle batches of gaze samples from eye tracker adapter.
        
        Args:
            samples: Batch of gaze samples from eye tracker.
        """
        if not samples:
            return
        
        # Update statistics
        self._stats["eye_samples_processed"] += len(samples)
        
        # Forward to signal processing layer
        self._signal_processing.add_samples(samples)
        
    
    def _on_eye_tracker_error(self, error: Exception) -> None:
        """
        Handle errors from eye tracker adapter.
        
        Args:
            error: Exception that occurred in eye tracker.
        """
        self._logger.system(
            "eye_tracker_error",
            {"error": str(error), "type": type(error).__name__},
            level="ERROR"
        )
        
        # Update connection state
        self._eye_tracker_connected = False
        
        # Publish status update
        self._publish(DomainEvent(
            event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
            payload=self.get_system_status(),
        ))
    
    def _on_window_features(self, features: WindowFeatures) -> None:
        """
        Handle new window features from Signal Processing.
        
        Args:
            features: Computed window features.
        """
        # TODO - log some stats

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
        # TODO - log some stats

        # Nothing to do in reactive mode
        if self._operation_mode == OperationMode.REACTIVE:
            return
        
        # In proactive mode, pass predicted features to Reactive Tool
        self._reactive_tool.add_features(predicted.to_window_features())
    
    def _on_user_state_estimate(self, estimate: UserStateEstimate) -> None:
        """
        Handle user state estimate from Reactive Tool.
        
        This is the trigger for feedback delivery. Feedback is only delivered
        when user state indicates it is appropriate.
        
        Args:
            estimate: User state estimate.
        """
        self._current_user_state = estimate

        self._logger.system(
            "user_state_estimate_updated",
            {
                "score": estimate.score,
                "contributing_features": estimate.contributing_features,
                "model_version": estimate.model_version,
                "model_type": estimate.model_type,
                "metadata": estimate.metadata,
            },
            level="DEBUG",
        )

        self._logger.experiment(
            "user_state_estimate_logged",
            {
                "score": estimate.score,
                "contributing_features": estimate.contributing_features,
                "model_version": estimate.model_version,
                "model_type": estimate.model_type,
                "metadata": estimate.metadata,
            },
            level="INFO",
        )

        # Attempt to deliver feedback if conditions are met
        self._try_deliver_feedback(force=False)
    
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

        # start processing layers if not already running
        self._signal_processing.start()

        # Start forecasting if in proactive mode
        if self._operation_mode == OperationMode.PROACTIVE:
            self._forecasting.enable()

        # Start reactive tool
        self._reactive_tool.start()

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

        # Stop processing layers and reset state
        self._signal_processing.stop()
        self._signal_processing.reset()

        # Stop forecasting
        self._forecasting.disable()

        # Stop reactive tool
        self._reactive_tool.stop()
        self._reactive_tool.reset()

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
        
        # Eye tracker adapter calls back to Runtime Controller
        if self._eye_tracker_adapter:
            self._eye_tracker_adapter.set_samples_callback(self._on_gaze_samples)
            self._eye_tracker_adapter.set_error_callback(self._on_eye_tracker_error)
        
        # Signal Processing calls back to Runtime Controller
        self._signal_processing.register_output_callback(
            self._on_window_features
        )

        # Forecasting Tool calls back to Runtime Controller
        self._forecasting.register_output_callback(
            self._on_predicted_features
        )

        # Reactive Tool calls back to Runtime Controller
        self._reactive_tool.register_output_callback(
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