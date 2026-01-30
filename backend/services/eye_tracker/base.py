"""
Base Eye Tracker Adapter Protocol

Defines the interface that all eye tracker adapters must implement.
This allows the RuntimeController to work with any eye tracker hardware
without direct dependencies on specific SDKs.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Dict, Any

from backend.types.eye_tracking import GazeSample


class AdapterState(Enum):
    """State of the eye tracker adapter."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


class EyeTrackerAdapter(ABC):
    """
    Abstract base class for eye tracker adapters.
    
    This interface allows the RuntimeController to communicate with
    different eye tracker implementations (real hardware or simulated)
    without knowing the specific implementation details.
    
    All adapters must implement this interface to ensure consistent
    behavior across different hardware types.
    """
    
    @abstractmethod
    async def connect(self, device_id: Optional[str] = None) -> bool:
        """
        Connect to an eye tracker device.
        
        Args:
            device_id: Optional device identifier. If None, connect to first available.
            
        Returns:
            True if connection successful, False otherwise.
            
        Note:
            Should transition state from DISCONNECTED -> CONNECTING -> CONNECTED on success.
            Should transition to ERROR on failure.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the eye tracker device.
        
        Note:
            Should ensure streaming is stopped before disconnecting.
            Should transition state to DISCONNECTED.
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if adapter is currently connected to a device.
        
        Returns:
            True if connected (state is CONNECTED or STREAMING), False otherwise.
        """
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the connected device.
        
        Returns:
            Dictionary containing device information:
                - device_id: Device identifier
                - model: Device model name
                - serial: Serial number
                - address: Network/connection address
                - sampling_rate_hz: Sampling rate in Hz
            Returns empty dict if not connected or info unavailable.
        """
        pass
    
    @abstractmethod
    def set_samples_callback(self, callback: Callable[[List[GazeSample]], None]) -> None:
        """
        Set callback function to receive batches of gaze samples.
        
        Args:
            callback: Function that will be called with batches of gaze samples.
                     The callback receives List[GazeSample] as parameter.
                     
        Note:
            The callback will be invoked from the asyncio event loop context,
            so it's safe to call async operations or update controller state.
        """
        pass
    
    @abstractmethod
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Set callback function to receive error notifications.
        
        Args:
            callback: Function that will be called when errors occur.
                     The callback receives Exception as parameter.
                     
        Note:
            The callback will be invoked from the asyncio event loop context.
        """
        pass
    
    @abstractmethod
    async def start_streaming(self) -> None:
        """
        Start streaming gaze data from the device.
        
        Note:
            Must be connected before starting streaming.
            Should transition state to STREAMING.
            Will call samples_callback with batches of GazeSample objects.
            
        Raises:
            RuntimeError: If not connected or already streaming.
        """
        pass
    
    @abstractmethod
    async def stop_streaming(self) -> None:
        """
        Stop streaming gaze data from the device.
        
        Note:
            Should gracefully cancel any streaming tasks.
            Should transition state to CONNECTED (still connected but not streaming).
        """
        pass
    
    @abstractmethod
    def get_state(self) -> AdapterState:
        """
        Get current adapter state.
        
        Returns:
            Current AdapterState.
        """
        pass
