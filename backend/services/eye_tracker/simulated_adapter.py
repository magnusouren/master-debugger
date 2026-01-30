"""
Simulated Eye Tracker Adapter

Provides a stub implementation of the eye tracker adapter for testing
and development without requiring real hardware.
"""
import asyncio
import time
from typing import Callable, List, Optional, Dict, Any

from backend.services.eye_tracker.base import EyeTrackerAdapter, AdapterState
from backend.types.eye_tracking import GazeSample
from backend.services.logger_service import get_logger


class SimulatedEyeTrackerAdapter(EyeTrackerAdapter):
    """
    Simulated eye tracker adapter for testing and development.
    
    This adapter generates synthetic gaze data at ~120 Hz and batches
    samples before sending them to the callback, mimicking the behavior
    of real hardware.
    """
    
    def __init__(
        self,
        sampling_rate_hz: float = 120.0,
        batch_size: int = 12,
        flush_interval_ms: int = 16,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        """
        Initialize simulated eye tracker adapter.
        
        Args:
            sampling_rate_hz: Simulated sampling rate in Hz.
            batch_size: Number of samples to batch before flushing.
            flush_interval_ms: Maximum time between flushes in milliseconds.
            loop: Asyncio event loop for callbacks.
        """
        self._sampling_rate_hz = sampling_rate_hz
        self._batch_size = batch_size
        self._flush_interval_ms = flush_interval_ms
        self._loop = loop or asyncio.get_event_loop()
        
        self._state = AdapterState.DISCONNECTED
        self._device_id = "simulated-eye-tracker-001"
        
        self._samples_callback: Optional[Callable[[List[GazeSample]], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        
        self._streaming_task: Optional[asyncio.Task] = None
        self._buffer: List[GazeSample] = []
        self._start_time: float = 0.0
        
        self._logger = get_logger()
    
    async def connect(self, device_id: Optional[str] = None) -> bool:
        """Connect to simulated eye tracker."""
        if self._state != AdapterState.DISCONNECTED:
            self._logger.system(
                "simulated_adapter_connect_invalid_state",
                {"current_state": self._state.value},
                level="WARNING"
            )
            return False
        
        self._state = AdapterState.CONNECTING
        
        # Simulate connection delay
        await asyncio.sleep(0.1)
        
        if device_id:
            self._device_id = device_id
        
        self._state = AdapterState.CONNECTED
        self._logger.system(
            "simulated_adapter_connected",
            {"device_id": self._device_id},
            level="INFO"
        )
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from simulated eye tracker."""
        if self._state == AdapterState.STREAMING:
            await self.stop_streaming()
        
        self._state = AdapterState.DISCONNECTED
        self._logger.system(
            "simulated_adapter_disconnected",
            {"device_id": self._device_id},
            level="INFO"
        )
    
    def is_connected(self) -> bool:
        """Check if simulated adapter is connected."""
        return self._state in (AdapterState.CONNECTED, AdapterState.STREAMING)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get simulated device information."""
        if not self.is_connected():
            return {}
        
        return {
            "device_id": self._device_id,
            "model": "Simulated Eye Tracker",
            "serial": "SIM-001",
            "address": "simulated://localhost",
            "sampling_rate_hz": self._sampling_rate_hz,
        }
    
    def set_samples_callback(self, callback: Callable[[List[GazeSample]], None]) -> None:
        """Set callback for sample batches."""
        self._samples_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for errors."""
        self._error_callback = callback
    
    async def start_streaming(self) -> None:
        """Start streaming simulated gaze data."""
        if self._state != AdapterState.CONNECTED:
            raise RuntimeError(
                f"Cannot start streaming: adapter is in {self._state.value} state"
            )
        
        if self._streaming_task and not self._streaming_task.done():
            self._logger.system(
                "simulated_adapter_already_streaming",
                {},
                level="WARNING"
            )
            return
        
        self._state = AdapterState.STREAMING
        self._start_time = time.time()
        self._buffer.clear()
        
        self._streaming_task = asyncio.create_task(self._stream_samples())
        
        self._logger.system(
            "simulated_adapter_streaming_started",
            {"device_id": self._device_id},
            level="INFO"
        )
    
    async def stop_streaming(self) -> None:
        """Stop streaming simulated gaze data."""
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None
        
        # Flush any remaining samples
        if self._buffer and self._samples_callback:
            self._samples_callback(self._buffer.copy())
        self._buffer.clear()
        
        if self._state == AdapterState.STREAMING:
            self._state = AdapterState.CONNECTED
            self._logger.system(
                "simulated_adapter_streaming_stopped",
                {"device_id": self._device_id},
                level="INFO"
            )
    
    def get_state(self) -> AdapterState:
        """Get current adapter state."""
        return self._state
    
    async def _stream_samples(self) -> None:
        """
        Background task that generates and batches simulated gaze samples.
        """
        sample_interval = 1.0 / self._sampling_rate_hz
        flush_interval = self._flush_interval_ms / 1000.0
        last_flush_time = time.time()
        
        try:
            while True:
                # Generate a sample
                current_time = time.time()
                sample = self._generate_sample(current_time)
                self._buffer.append(sample)
                
                # Flush if batch is full or interval elapsed
                should_flush = (
                    len(self._buffer) >= self._batch_size or
                    (current_time - last_flush_time) >= flush_interval
                )
                
                if should_flush and self._samples_callback:
                    batch = self._buffer.copy()
                    self._buffer.clear()
                    
                    # Call callback in asyncio context
                    self._samples_callback(batch)
                    last_flush_time = current_time
                
                # Wait until next sample time
                await asyncio.sleep(sample_interval)
                
        except asyncio.CancelledError:
            # Clean shutdown
            raise
        except Exception as e:
            self._logger.system(
                "simulated_adapter_streaming_error",
                {"error": str(e)},
                level="ERROR"
            )
            self._state = AdapterState.ERROR
            if self._error_callback:
                self._error_callback(e)
            raise
    
    def _generate_sample(self, timestamp: float) -> GazeSample:
        """
        Generate a synthetic gaze sample.
        
        Args:
            timestamp: Current timestamp.
            
        Returns:
            Simulated GazeSample.
        """
        # Simple simulation: gaze moves in a circular pattern
        elapsed = timestamp - self._start_time
        
        # Normalized coordinates (0-1 range)
        # Simulate smooth eye movements with some variation
        base_x = 0.5 + 0.3 * (elapsed % 3.0) / 3.0  # Drift right
        base_y = 0.5 + 0.2 * ((elapsed % 2.0) / 2.0)  # Drift down
        
        # Add small random variation (micro-saccades)
        import random
        noise = 0.02
        
        left_x = base_x + random.uniform(-noise, noise)
        left_y = base_y + random.uniform(-noise, noise)
        right_x = base_x + random.uniform(-noise, noise)
        right_y = base_y + random.uniform(-noise, noise)
        
        # Simulate pupil diameter (2-8 mm typical range)
        pupil_base = 4.0 + 1.0 * (elapsed % 5.0) / 5.0
        left_pupil = pupil_base + random.uniform(-0.2, 0.2)
        right_pupil = pupil_base + random.uniform(-0.2, 0.2)
        
        return GazeSample(
            timestamp=timestamp,
            left_eye_x=left_x,
            left_eye_y=left_y,
            right_eye_x=right_x,
            right_eye_y=right_y,
            left_pupil_diameter=left_pupil,
            right_pupil_diameter=right_pupil,
            left_eye_valid=True,
            right_eye_valid=True,
            raw_data={"source": "simulated"}
        )
