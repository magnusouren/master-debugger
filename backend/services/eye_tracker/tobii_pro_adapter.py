"""
Tobii Pro Eye Tracker Adapter

Provides integration with Tobii Pro SDK for real eye tracker hardware.
Uses lazy imports to avoid hard dependency on the SDK.
"""
import asyncio
import time
import threading
from typing import Callable, List, Optional, Dict, Any

from backend.services.eye_tracker.base import EyeTrackerAdapter, AdapterState
from backend.types.eye_tracking import GazeSample
from backend.services.logger_service import get_logger


class TobiiProEyeTrackerAdapter(EyeTrackerAdapter):
    """
    COMPLETELY AI GENERATED FILE. MAY CONTAIN BUGS.

    Eye tracker adapter for Tobii Pro SDK hardware.
    
    This adapter connects to real Tobii eye trackers, subscribes to gaze data,
    and batches samples efficiently before forwarding to the controller.
    
    Key features:
    - Lazy SDK import (no hard dependency)
    - Thread-safe callback handling from Tobii SDK
    - Efficient sample batching with time-based flushing
    - Robust error handling and state management
    """
    
    def __init__(
        self,
        batch_size: int = 12,
        flush_interval_ms: int = 16,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        """
        Initialize Tobii Pro eye tracker adapter.
        
        Args:
            batch_size: Number of samples to batch before flushing.
            flush_interval_ms: Maximum time between flushes in milliseconds.
            loop: Asyncio event loop for thread-safe callbacks.
        """
        self._batch_size = batch_size
        self._flush_interval_ms = flush_interval_ms
        self._loop = loop or asyncio.get_event_loop()
        
        self._state = AdapterState.DISCONNECTED
        self._device = None
        self._device_info: Dict[str, Any] = {}
        
        self._samples_callback: Optional[Callable[[List[GazeSample]], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        
        self._buffer: List[GazeSample] = []
        self._buffer_lock = threading.Lock()
        
        self._flush_task: Optional[asyncio.Task] = None
        self._subscribed = False
        
        self._logger = get_logger()
        
        # Tobii SDK module (lazy loaded)
        self._tobii_research = None
    
    async def connect(self, device_id: Optional[str] = None) -> bool:
        """Connect to Tobii Pro eye tracker."""
        if self._state != AdapterState.DISCONNECTED:
            self._logger.system(
                "tobii_adapter_connect_invalid_state",
                {"current_state": self._state.value},
                level="WARNING"
            )
            return False
        
        self._state = AdapterState.CONNECTING
        
        try:
            # Lazy import Tobii SDK
            try:
                import tobii_research as tr
                self._tobii_research = tr
            except ImportError as e:
                error_msg = (
                    "Tobii Pro SDK not installed. "
                    "Install with: pip install tobii-research"
                )
                self._logger.system(
                    "tobii_sdk_not_found",
                    {"error": error_msg},
                    level="ERROR"
                )
                self._state = AdapterState.ERROR
                raise RuntimeError(error_msg) from e
            
            # Find eye trackers
            self._logger.system(
                "tobii_adapter_discovering_devices",
                {},
                level="DEBUG"
            )
            
            found_eyetrackers = tr.find_all_eyetrackers()
            
            if not found_eyetrackers:
                self._logger.system(
                    "tobii_no_devices_found",
                    {},
                    level="WARNING"
                )
                self._state = AdapterState.DISCONNECTED
                return False
            
            # Select device
            if device_id:
                # Try to match by serial number, address, or device name
                self._device = None
                for et in found_eyetrackers:
                    if (et.serial_number == device_id or
                        et.address == device_id or
                        et.device_name == device_id):
                        self._device = et
                        break
                
                if not self._device:
                    self._logger.system(
                        "tobii_device_not_found",
                        {"device_id": device_id},
                        level="WARNING"
                    )
                    self._state = AdapterState.DISCONNECTED
                    return False
            else:
                # Use first available device
                self._device = found_eyetrackers[0]
            
            # Store device info
            self._device_info = {
                "device_id": self._device.serial_number,
                "model": self._device.model,
                "serial": self._device.serial_number,
                "address": self._device.address,
                "device_name": self._device.device_name,
                "firmware_version": getattr(self._device, "firmware_version", "unknown"),
            }
            
            # Get sampling rate if available
            try:
                gaze_output_frequency = self._device.get_gaze_output_frequency()
                self._device_info["sampling_rate_hz"] = gaze_output_frequency
            except Exception as e:
                self._logger.system(
                    "tobii_unable_to_get_sampling_rate",
                    {"error": str(e)},
                    level="DEBUG"
                )
                self._device_info["sampling_rate_hz"] = 120.0  # Default assumption
            
            self._state = AdapterState.CONNECTED
            self._logger.system(
                "tobii_adapter_connected",
                self._device_info,
                level="INFO"
            )
            return True
            
        except Exception as e:
            self._logger.system(
                "tobii_adapter_connect_error",
                {"error": str(e)},
                level="ERROR"
            )
            self._state = AdapterState.ERROR
            if self._error_callback:
                self._loop.call_soon_threadsafe(self._error_callback, e)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Tobii Pro eye tracker."""
        if self._state == AdapterState.STREAMING:
            await self.stop_streaming()
        
        self._device = None
        self._device_info = {}
        self._state = AdapterState.DISCONNECTED
        
        self._logger.system(
            "tobii_adapter_disconnected",
            {},
            level="INFO"
        )
    
    def is_connected(self) -> bool:
        """Check if Tobii adapter is connected."""
        return self._state in (AdapterState.CONNECTED, AdapterState.STREAMING)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get Tobii device information."""
        if not self.is_connected():
            return {}
        return self._device_info.copy()
    
    def set_samples_callback(self, callback: Callable[[List[GazeSample]], None]) -> None:
        """Set callback for sample batches."""
        self._samples_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for errors."""
        self._error_callback = callback
    
    async def start_streaming(self) -> None:
        """Start streaming gaze data from Tobii eye tracker."""
        if self._state != AdapterState.CONNECTED:
            raise RuntimeError(
                f"Cannot start streaming: adapter is in {self._state.value} state"
            )
        
        if not self._device:
            raise RuntimeError("No device connected")
        
        if self._subscribed:
            self._logger.system(
                "tobii_adapter_already_streaming",
                {},
                level="WARNING"
            )
            return
        
        try:
            # Subscribe to gaze data
            self._device.subscribe_to(
                self._tobii_research.EYETRACKER_GAZE_DATA,
                self._on_gaze_data,
                as_dictionary=True
            )
            self._subscribed = True
            
            # Start periodic flush task
            self._flush_task = asyncio.create_task(self._periodic_flush())
            
            self._state = AdapterState.STREAMING
            self._logger.system(
                "tobii_adapter_streaming_started",
                {"device": self._device_info.get("device_id", "unknown")},
                level="INFO"
            )
            
        except Exception as e:
            self._logger.system(
                "tobii_adapter_start_streaming_error",
                {"error": str(e)},
                level="ERROR"
            )
            self._state = AdapterState.ERROR
            if self._error_callback:
                self._loop.call_soon_threadsafe(self._error_callback, e)
            raise
    
    async def stop_streaming(self) -> None:
        """Stop streaming gaze data from Tobii eye tracker."""
        # Cancel periodic flush
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        
        # Unsubscribe from Tobii stream
        if self._subscribed and self._device:
            try:
                self._device.unsubscribe_from(
                    self._tobii_research.EYETRACKER_GAZE_DATA,
                    self._on_gaze_data
                )
                self._subscribed = False
            except Exception as e:
                self._logger.system(
                    "tobii_adapter_unsubscribe_error",
                    {"error": str(e)},
                    level="WARNING"
                )
        
        # Flush remaining samples
        with self._buffer_lock:
            if self._buffer and self._samples_callback:
                batch = self._buffer.copy()
                self._buffer.clear()
                self._loop.call_soon_threadsafe(self._samples_callback, batch)
        
        if self._state == AdapterState.STREAMING:
            self._state = AdapterState.CONNECTED
            self._logger.system(
                "tobii_adapter_streaming_stopped",
                {},
                level="INFO"
            )
    
    def get_state(self) -> AdapterState:
        """Get current adapter state."""
        return self._state
    
    def _on_gaze_data(self, gaze_data: Dict[str, Any]) -> None:
        """
        Callback from Tobii SDK with gaze data (runs on SDK thread).
        
        Args:
            gaze_data: Dictionary containing gaze data from Tobii SDK.
        """
        try:
            # Convert Tobii data to GazeSample
            sample = self._convert_tobii_sample(gaze_data)
            
            # Add to buffer (thread-safe)
            should_flush = False
            with self._buffer_lock:
                self._buffer.append(sample)
                if len(self._buffer) >= self._batch_size:
                    should_flush = True
            
            # Flush if batch is full
            if should_flush:
                self._flush_buffer()
                
        except Exception as e:
            self._logger.system(
                "tobii_adapter_gaze_callback_error",
                {"error": str(e)},
                level="ERROR"
            )
            self._state = AdapterState.ERROR
            if self._error_callback:
                self._loop.call_soon_threadsafe(self._error_callback, e)
    
    def _convert_tobii_sample(self, gaze_data: Dict[str, Any]) -> GazeSample:
        """
        Convert Tobii SDK gaze data to GazeSample.

        Args:
            gaze_data: Gaze data dictionary from Tobii SDK.

        Returns:
            GazeSample object.
        """
        # Use device timestamp if available (microseconds), otherwise system time
        # Device timestamps are more precise, especially on Windows where time.time()
        # has ~15.6ms resolution which is too coarse for 120Hz data
        device_ts = gaze_data.get("device_time_stamp")
        if device_ts is not None:
            # Convert from microseconds to seconds
            timestamp = device_ts / 1_000_000.0
        else:
            timestamp = time.time()
        
        # Extract left eye data
        left_eye = gaze_data.get("left_gaze_point_on_display_area", (None, None))
        left_pupil = gaze_data.get("left_pupil_diameter", None)
        left_valid = gaze_data.get("left_gaze_point_validity", False)
        
        # Extract right eye data
        right_eye = gaze_data.get("right_gaze_point_on_display_area", (None, None))
        right_pupil = gaze_data.get("right_pupil_diameter", None)
        right_valid = gaze_data.get("right_gaze_point_validity", False)
        
        return GazeSample(
            timestamp=timestamp,
            left_eye_x=left_eye[0] if isinstance(left_eye, (tuple, list)) and len(left_eye) > 0 else None,
            left_eye_y=left_eye[1] if isinstance(left_eye, (tuple, list)) and len(left_eye) > 1 else None,
            right_eye_x=right_eye[0] if isinstance(right_eye, (tuple, list)) and len(right_eye) > 0 else None,
            right_eye_y=right_eye[1] if isinstance(right_eye, (tuple, list)) and len(right_eye) > 1 else None,
            left_pupil_diameter=left_pupil,
            right_pupil_diameter=right_pupil,
            left_eye_valid=left_valid,
            right_eye_valid=right_valid,
            raw_data=gaze_data
        )
    
    def _flush_buffer(self) -> None:
        """Flush buffered samples to callback (thread-safe)."""
        with self._buffer_lock:
            if not self._buffer or not self._samples_callback:
                return
            
            batch = self._buffer.copy()
            self._buffer.clear()
        
        # Forward to callback in asyncio loop
        self._loop.call_soon_threadsafe(self._samples_callback, batch)
    
    async def _periodic_flush(self) -> None:
        """
        Periodic task to flush buffer even if not full.
        
        This ensures samples are delivered in a timely manner even if
        the batch size is not reached (e.g., during low-frequency events).
        """
        flush_interval = self._flush_interval_ms / 1000.0
        
        try:
            while True:
                await asyncio.sleep(flush_interval)
                self._flush_buffer()
                
        except asyncio.CancelledError:
            # Clean shutdown
            raise
        except Exception as e:
            self._logger.system(
                "tobii_adapter_periodic_flush_error",
                {"error": str(e)},
                level="ERROR"
            )
            self._state = AdapterState.ERROR
            if self._error_callback:
                self._loop.call_soon_threadsafe(self._error_callback, e)
            raise
