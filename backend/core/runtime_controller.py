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
        self._logger.set_experiment_mode(self._config.controller.operation_mode.value)
        
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
        

        # ID and timing tracking
        self._window_counter: int = 0
        self._estimate_counter: int = 0
        self._experiment_start_time: Optional[float] = None
        self._id_prefix: str = self._session_id or "session"
        
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
        self._status = SystemStatus.INITIALIZING
        self._logger.system("runtime_controller_initializing", {}, level="DEBUG")
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
        
        self._status = SystemStatus.READY
        self._logger.system("runtime_controller_ready", self.get_system_status(), level="INFO")

        return True
    
    async def shutdown(self) -> None:
        """Shutdown all system components gracefully."""
        self._logger.system("runtime_controller_shutdown", {"final_stats": self._stats}, level="INFO")

        if self._experiment_is_active:
            await self.end_experiment()
        
        self._status = SystemStatus.STOPPED

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
            {"operation_mode": self._operation_mode.value},
            level="INFO",
        )
    
    def set_operation_mode(self, mode: OperationMode) -> None:
        """
        Set the system operation mode.
        
        """
        old_mode = self._operation_mode

        self._operation_mode = mode
        self._logger.set_experiment_mode(mode.value)

        # Keep the stored configuration in sync with the current operation mode
        if self._config is not None and getattr(self._config, "controller", None) is not None:
            self._config.controller.operation_mode = mode

        # Enable or disable layers based on mode
        if mode == OperationMode.PROACTIVE:
            self._forecasting.reset()  # Clear any old state when switching to proactive
            self._forecasting.enable()
            self._logger.system(
                "forecasting_enabled_for_proactive_mode",
                {},
                level="DEBUG",
            )
        else:
            self._forecasting.disable()
            self._logger.system(
                "forecasting_disabled_for_non_proactive_mode",
                {},
                level="DEBUG",
            )
        
        self._logger.system(
            "operation_mode_changed",
            {"new_mode": self._operation_mode.name, "old_mode": old_mode.name},
            level="INFO",
        )

        self._logger.experiment(
            "operation_mode_changed",
            {"new_mode": self._operation_mode.name, "old_mode": old_mode.name},
            level="INFO",

        )

        # Reconfigure layers as needed using the updated configuration
        if self._config is not None:
            self.configure(self._config)

        # Publish status update
        self._publish(DomainEvent(
            event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
            payload=self.get_system_status(),
        ))
    
    def get_system_status(self) -> SystemStatusMessage:
        """
        Get detailed system status message.
        
        Returns:
            SystemStatusMessage with current status details.
        """
        return SystemStatusMessage(
            status=self._status.value,
            timestamp=datetime.now(timezone.utc).timestamp(),
            eye_tracker_model=self.get_eye_tracker_model(),
            operation_mode=self._operation_mode.value,
            eye_samples_processed=self._stats["eye_samples_processed"],
            code_window_samples_processed=self._stats["code_window_samples_processed"],
            feedback_generated=self._stats["feedback_generated"],
            llm_model=self._feedback_layer.get_llm_client().get_model_name() if self._feedback_layer.get_llm_client() else None,
            feedback_cooldown_left_s=int(self.get_feedback_cooldown_remaining()),
            experiment_active=self._experiment_is_active,
            experiment_id=self._experiment_id,
            participant_id=self._participant_id,
            user_state_score=self._current_user_state.score.score if self._current_user_state else None,
        )
    
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
    
    def get_eye_tracker_model(self) -> Optional[str]:
        """
        Get model name of connected eye tracker.

        Returns:
            Eye tracker model if connected, otherwise None.
        """
        if not self._eye_tracker_connected or not self._eye_tracker_adapter:
            return None

        device_info = self._eye_tracker_adapter.get_device_info()
        model = device_info.get("model") if device_info else None
        return str(model) if model else None
    
    # --- VS Code Communication ---
    
    async def handle_context_update(self, context: CodeContext) -> None:
        if not self._experiment_is_active:
            return
        
        if self._operation_mode in (OperationMode.CONTROL, OperationMode.QUESTIONNAIRE):
            self._logger.system(
                "context_update_ignored_in_non_feedback_mode",
                {"mode": self._operation_mode.value},
                level="DEBUG",
            )
            return

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

        self._stats["code_window_samples_processed"] += 1 

    def manual_send_feedback(self) -> bool:
        """
        Manual trigger to deliver the latest available feedback immediately.

        Returns:
            True if something was delivered, False otherwise.
        """
        self._logger.system(
            "manual_feedback_triggered",
            {
                "pending_feedback_version": self._pending_feedback_version,
                "last_delivered_version": self._last_delivered_version,
                "cooldown_remaining": self.get_feedback_cooldown_remaining(),
            },
        )

        return self._try_deliver_feedback(force=True)
    
    async def handle_feedback_interaction(
        self, interaction: FeedbackInteraction
    ) -> bool:
        """
        Handle user interaction with feedback.
        
        Interaction types:
        - presented: Feedback was shown to user
        - accepted: User accepted to see feedback details
        - rejected: User rejected seeing the feedback
        - highlighted: User clicked to highlight in code
        - dismissed: User dismissed the shown feedback
        
        Args:
            interaction: User interaction data.
        """

        # Map interaction types to log categories
        category_map = {
            "presented": "feedback_presented_to_user",
            "accepted": "feedback_accepted_by_user",
            "rejected": "feedback_rejected_by_user",
            "highlighted": "feedback_highlighted_in_code",
            "dismissed": "feedback_dismissed_by_user",
            "done": "feedback_marked_done_by_user",
        }
        
        category_msg = category_map.get(
            interaction.interaction_type,
            f"feedback_interaction_unknown_type: {interaction.interaction_type}"
        )
        
        self._logger.system(
            category_msg,
            {
                "feedback_id": interaction.feedback_id,
                "action_taken": interaction.interaction_type,
            },
            level="INFO",
        )
        self._logger.experiment(
            category_msg,
            {
                "feedback_id": interaction.feedback_id,
                "action_taken": interaction.interaction_type,
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

    def _next_window_id(self) -> str:
        self._window_counter += 1
        return f"{self._id_prefix}-w{self._window_counter:06d}"

    def _next_estimate_id(self) -> str:
        self._estimate_counter += 1
        return f"{self._id_prefix}-e{self._estimate_counter:06d}"

    def _experiment_time_sec(self, timestamp: float) -> Optional[float]:
        if self._experiment_start_time is None:
            return None
        return max(0.0, timestamp - self._experiment_start_time)
    
    # --- Feedback Control ---
    
    def _can_start_feedback_generation(self) -> bool:
        """
        Determine if feedback generation can be started.
        
        This checks preconditions for starting a new feedback generation task.
        Actual delivery decisions (cooldown, threshold) are checked at delivery time.
        
        Returns:
            True if feedback generation can be started.
        """
        if self._status != SystemStatus.RUNNING:
            return False

        if self._operation_mode in (OperationMode.CONTROL, OperationMode.QUESTIONNAIRE):
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
        if self._status != SystemStatus.RUNNING:
            return False

        if self._operation_mode in (OperationMode.CONTROL, OperationMode.QUESTIONNAIRE):
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
                level="INFO",
            )
            self._logger.experiment(
                "feedback_delivery_threshold_met",
                {
                    "score": self._current_user_state.score.score,
                    "threshold": threshold,
                    "pending_version": self._pending_feedback_version,
                },
                level="INFO",
            )
        return True
    
    def _try_deliver_feedback(self, force: bool = False) -> bool:
        """
        Attempt to deliver the latest pending feedback.

        If force=True, bypass threshold/cooldown checks and deliver the latest
        pending feedback if available.
        """
        if force:
            if self._status != SystemStatus.RUNNING:
                return False

            if self._operation_mode in (OperationMode.CONTROL, OperationMode.QUESTIONNAIRE):
                return False

            if self._pending_feedback is None:
                return False

            feedback = self._pending_feedback
            version = self._pending_feedback_version

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
                "trigger": event_meta["trigger"],
                "recipient_id": recipient_id,
                "feedback_version": version,
                "item_count": len(feedback.items),
                "item_ids": [item.metadata.feedback_id for item in feedback.items],
            },
            level="INFO",
        )

        self._logger.experiment(
            "feedback_delivered",
            {
                "trigger": event_meta["trigger"],
                "feedback_version": version,
                "item_ids": [item.metadata.feedback_id for item in feedback.items],
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
            )

            for item in feedback.items:
                self._logger.feedback(
                    "feedback_item_generated",
                    item
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
        
        last_feedback_time = self._last_feedback_time

        if last_feedback_time == 0.0:
            last_feedback_time = self._stats.get("session_start", current_time)
        
        elapsed = current_time - last_feedback_time

        remaining = max(0.0, cooldown - elapsed)
        return remaining

    def set_feedback_cooldown(self, cooldown_seconds: float) -> None:
        """
        Set the feedback cooldown duration.
        
        Args:
            cooldown_seconds: New cooldown duration in seconds.
        """
        self._last_feedback_time = asyncio.get_event_loop().time() # reset cooldown timer on change
        self._config.controller.feedback_cooldown_seconds = cooldown_seconds
        
        self._logger.system(
            "feedback_cooldown_changed",
            {"new_cooldown_seconds": cooldown_seconds},
            level="INFO",
        )
        
        # Publish status update
        self._publish(DomainEvent(
            event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
            payload=self.get_system_status(),
        ))

    # --- Baseline Calibration ---

    async def _run_baseline_calibration(self, duration_seconds: float) -> None:
        """
        Automatic baseline calibration sequence.

        Waits 5 seconds, then records baseline for the amount of seconds specified.
        """
        try:
            # Wait 5 seconds for things to settle
            self._logger.system(
                "baseline_calibration_waiting",
                {
                    "wait_seconds": 5, 
                    "participant_id": self._participant_id,
                    "experiment_id": self._experiment_id,
                },
                level="INFO"
            )
            original_cooldown = self._config.controller.feedback_cooldown_seconds
            self.set_feedback_cooldown(duration_seconds + original_cooldown)

            await asyncio.sleep(5)

            # Check if experiment is still active
            if not self._experiment_is_active:
                return

            # Start baseline recording
            self._status = SystemStatus.CALIBRATING
            self._reactive_tool.start_baseline_recording(self._participant_id)

            # Record for the specified duration
            self._logger.system(
                "baseline_calibration_recording",
                {
                    "duration_seconds": duration_seconds, 
                    "participant_id": self._participant_id,
                    "experiment_id": self._experiment_id,   
                },
                level="INFO"
            )

            self._logger.experiment(
                "baseline_calibration_recording",
                {
                    "duration_seconds": duration_seconds, 
                },
                level="INFO"
            )
            await asyncio.sleep(duration_seconds)

            # Check if experiment is still active
            if not self._experiment_is_active:
                return

            # Stop baseline recording
            baseline = self._reactive_tool.stop_baseline_recording(self._participant_id)

            if baseline:
                self._logger.system(
                    "baseline_calibration_completed",
                    {
                        "metrics": {k: {"mean": round(v.mean, 4), "std": round(v.std, 4)}
                                   for k, v in baseline.metrics.items()},
                    },
                    level="INFO"
                )

                self._logger.experiment(
                    "baseline_calibration_completed",
                    {
                        "metrics": {k: {"mean": round(v.mean, 4), "std": round(v.std, 4)}
                                   for k, v in baseline.metrics.items()},
                    },
                    level="INFO"
                )

                if self._experiment_is_active:
                    self.set_feedback_cooldown(original_cooldown)
                    self._status = SystemStatus.RUNNING
                    self._publish(DomainEvent(
                        event_type=DomainEventType.CODE_CONTEXT_NEEDED,
                    ))
                else:
                    self._status = SystemStatus.READY
            else:
                self._logger.system(
                    "baseline_calibration_failed",
                    {
                        "reason": "insufficient_data",
                        "participant_id": self._participant_id, 
                        "experiment_id": self._experiment_id,
                    },
                    level="ERROR"
                )
                self._status = SystemStatus.ERROR

        except asyncio.CancelledError:
            self._logger.system(
                "baseline_calibration_cancelled",
                {"participant_id": self._participant_id, "experiment_id": self._experiment_id},
                level="INFO"
            )
            self._status = SystemStatus.ERROR

        except Exception as e:
            self._logger.system(
                "baseline_calibration_error",
                {"participant_id": self._participant_id, "experiment_id": self._experiment_id, "error": str(e)},
                level="ERROR"
            )
            self._status = SystemStatus.ERROR

    def start_baseline_recording(self, participant_id: str) -> None:
        """
        Start recording baseline metrics for a participant.

        Call this when the participant begins the baseline task (e.g., reading simple text).

        Args:
            participant_id: Unique identifier for the participant.
        """
        self._reactive_tool.start_baseline_recording(participant_id)

    def stop_baseline_recording(self, participant_id: str):
        """
        Stop recording and compute baseline statistics.

        Args:
            participant_id: Unique identifier for the participant.

        Returns:
            ParticipantBaseline object with computed statistics, or None if insufficient data.
        """
        return self._reactive_tool.stop_baseline_recording(participant_id)

    def clear_baseline(self) -> None:
        """Clear the current baseline (revert to static thresholds)."""
        self._reactive_tool.clear_baseline()

    def has_baseline(self) -> bool:
        """Check if a valid baseline is loaded."""
        return self._reactive_tool.has_baseline()

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
        features.window_id = features.window_id or self._next_window_id()
        features.is_predicted = False
        features.forecast_id = None

        feature_window_size = features.window_end - features.window_start
        self._logger.features(
            "observed_feature_window_logged",
            {
                "window_id": features.window_id,
                "window_start": features.window_start,
                "window_end": features.window_end,
                "feature_window_size": feature_window_size,
                "features": features.features,
                "valid_sample_ratio": features.valid_sample_ratio,
                "using_baseline": self._reactive_tool.has_baseline(),
                "mode": self._operation_mode.value,
                "is_predicted": False,
            },
            level="INFO",
        )

        if self._operation_mode in (
            OperationMode.REACTIVE,
            OperationMode.CONTROL,
            OperationMode.QUESTIONNAIRE,
        ):
            # Baseline: observed features -> reactive
            self._reactive_tool.add_features(features)
            return
        
        if self._operation_mode == OperationMode.PROACTIVE:
            # For logging
            self._reactive_tool.add_features(features)
            # Proactive: observed features -> forecasting
            self._forecasting.add_features(features)
        
        # During baseline recording in proactive mode, feed observed windows
        # to reactive so baseline statistics are computed from real signal data.
        if self._reactive_tool.is_recording_baseline():
            self._reactive_tool.add_features(features)

    
    def _on_predicted_features(self, predicted: PredictedFeatures) -> None:
        """
        Handle predicted features from Forecasting Tool.
        
        Args:
            predicted: Predicted features.
        """

        # Nothing to do in reactive mode
        if self._operation_mode in (
            OperationMode.REACTIVE,
            OperationMode.CONTROL,
            OperationMode.QUESTIONNAIRE,
        ):
            return

        # Don't mix forecasted windows into baseline computation.
        if self._reactive_tool.is_recording_baseline():
            return

        if not predicted.window_id:
            predicted.window_id = self._next_window_id()

        if not predicted.forecast_id:
            predicted.forecast_id = "forecast-unknown"

        self._logger.features(
            "predicted_feature_window_logged",
            {
                "window_id": predicted.window_id,
                "forecast_id": predicted.forecast_id,
                "target_window_start": predicted.target_window_start,
                "target_window_end": predicted.target_window_end,
                "target_time": predicted.target_window_start,
                "predicted_features": predicted.features,
                "prediction_horizon_seconds": predicted.horizon_seconds,
                "is_predicted": True,
            },
            level="INFO",
        )

        # Unified path:
        # Signal -> Forecasting(predicted component values) -> Reactive(score + baseline)
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

        estimate.estimate_id = estimate.estimate_id or self._next_estimate_id()

        source_window_id = estimate.source_window_id or estimate.metadata.get("source_window_id")
        if not source_window_id:
            window_ids = estimate.metadata.get("window_ids") if estimate.metadata else []
            if window_ids:
                source_window_id = window_ids[-1]

        forecast_id = estimate.forecast_id
        if forecast_id is None and estimate.metadata:
            forecast_id = estimate.metadata.get("forecast_id")
        source_type = estimate.source_type or (estimate.metadata.get("source_type") if estimate.metadata else None) or "observed_features"
        experiment_time_sec = self._experiment_time_sec(estimate.timestamp)

        log_payload = {
            "estimate_id": estimate.estimate_id,
            "score": estimate.score.score,
            "confidence": estimate.score.confidence,
            "source_type": source_type,
            "source_window_id": source_window_id,
            "forecast_id": forecast_id,
            "experiment_time_sec": experiment_time_sec,
            "contributing_features": estimate.contributing_features,
            "metadata": estimate.metadata,
            "window_start": estimate.metadata.get("window_start") if estimate.metadata else None,
            "window_end": estimate.metadata.get("window_end") if estimate.metadata else None,
        }

        self._logger.system(
            "user_state_estimate_updated",
            log_payload,
            level="DEBUG",
        )



        if self._operation_mode == OperationMode.PROACTIVE and source_type == "observed_features":
            self._logger.experiment(
                "user_state_estimate_logged",
                {**log_payload, "target_time_sec": estimate.metadata.get("target_time_sec") if estimate.metadata else None},
                level="INFO",
            )
            return

        # Control mode does not deliver feedback, but we still want to log the user state estimates for analysis
        if self._operation_mode in (OperationMode.CONTROL, OperationMode.QUESTIONNAIRE):
            return

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

        self._logger.set_start_time()

        # Start streaming from eye tracker if not already started
        if self._eye_tracker_adapter and self._eye_tracker_adapter.is_connected():
            await self._eye_tracker_adapter.start_streaming()
        else:
            self._logger.system(
                "eye_tracker_not_connected_on_experiment_start",
                {
                    "message": "Eye tracker is not connected at experiment start. Eye tracking data will not be available.",
                },
                level="ERROR"
            )
            return self.get_system_status()
            # TODO - handle this case with the UI

        self._experiment_id = experiment_id
        self._participant_id = participant_id
        self._experiment_is_active = True
        self._session_id = f"{participant_id}_{experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self._id_prefix = self._session_id or self._id_prefix
        self._window_counter = 0
        self._estimate_counter = 0
        self._experiment_start_time = datetime.now(timezone.utc).timestamp()

        # start processing layers if not already running
        self._signal_processing.start()

        # Start forecasting if in proactive mode
        if self._operation_mode == OperationMode.PROACTIVE:
            self._forecasting.enable()

        # Start reactive tool
        self._reactive_tool.start()

        # Schedule automatic baseline recording
        # Wait 5 seconds, then record baseline calibration for the amount of seconds specified in config
        asyncio.create_task(self._run_baseline_calibration(self._config.controller.calibration_duration_seconds))

        self._logger.system(
            "experiment_started",
            {
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id,
                "mode": self._operation_mode.value,
            },
            level="INFO",
        )

        self._logger.experiment(
            "experiment_started",
            {
                "experiment_id": self._experiment_id,
                "participant_id": self._participant_id,
                "session_id": self._session_id,
                "mode": self._operation_mode.value,
            },
            level="INFO",
        )

        self._status = SystemStatus.RUNNING

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

        # Publish initial system status update after experiment start
        self._publish(DomainEvent(
            event_type=DomainEventType.SYSTEM_STATUS_UPDATED,
            payload=self.get_system_status(),
        ))

        return self.get_system_status()

    async def end_experiment(self) -> SystemStatusMessage:
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
        self.export_system_logs()
        self.export_feedback_logs()
        self.export_feature_logs()
        self._status = SystemStatus.STOPPED

        # Stop processing layers and reset state
        self._signal_processing.stop()
        self._signal_processing.reset()

        # Stop forecasting
        self._forecasting.disable()

        # Stop reactive tool
        self._reactive_tool.stop()
        self._reactive_tool.reset()

        # Stop eye tracker streaming (but keep connection for potential future sessions)
        await self._eye_tracker_adapter.stop_streaming()

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
        self._experiment_start_time = None

        return self.get_system_status()
    
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
    
    def export_system_logs(self) -> bool:
        """
        Export system logs to file.
        
        Returns:
            True if export successful.
        """
        if self._experiment_id is None or self._participant_id is None:
            self._logger.system(
                "export_system_logs_no_experiment",
                {},
                level="WARNING",
            )
            return False

        filepath = f"logs/system/system_{self._session_id}.csv"
        success = self._logger.export_system_logs(filepath)
        return success

    def export_feedback_logs(self) -> bool:
        """
        Export feedback logs to file.
        
        Returns:
            True if export successful.
        """
        if self._experiment_id is None or self._participant_id is None:
            self._logger.system(
                "export_feedback_logs_no_experiment",
                {},
                level="WARNING",
            )
            return False

        filepath = f"logs/feedback/feedback_{self._session_id}.csv"
        success = self._logger.export_feedback_logs(filepath)
        return success

    def export_feature_logs(self) -> bool:
        """
        Export feature stream logs to file.

        Returns:
            True if export successful.
        """
        if self._experiment_id is None or self._participant_id is None:
            self._logger.system(
                "export_feature_logs_no_experiment",
                {},
                level="WARNING",
            )
            return False

        filepath = f"logs/features/features_{self._session_id}.csv"
        success = self._logger.export_feature_logs(filepath)
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
    
    async def _run_main_loop(self) -> None:
        """Main processing loop."""

        self._logger.system("runtime_controller_main_loop_started", {}, level="DEBUG")
        while self._status != SystemStatus.DISCONNECTED:
            await asyncio.sleep(1)  # Main loop tick
            
            # Update statistics
            self._update_statistics()

            # Publish system status periodically
            self._broadcast_system_status()

            # Request code context if we don't have one and experiment is active (and we're close to cooldown expiring)
            if self._status == SystemStatus.RUNNING and self._current_code_context is None and self.get_feedback_cooldown_remaining() <= 10.0:
                self._publish(DomainEvent(
                    event_type=DomainEventType.CODE_CONTEXT_NEEDED,
                ))

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
