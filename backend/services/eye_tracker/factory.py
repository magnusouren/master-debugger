"""
Eye Tracker Adapter Factory

Creates and configures eye tracker adapters based on system configuration.
"""
import asyncio
from typing import Optional

from backend.services.eye_tracker.base import EyeTrackerAdapter
from backend.services.eye_tracker.simulated_adapter import SimulatedEyeTrackerAdapter
from backend.services.eye_tracker.tobii_pro_adapter import TobiiProEyeTrackerAdapter
from backend.types.config import SystemConfig
from backend.services.logger_service import get_logger


def create_eye_tracker_adapter(
    config: SystemConfig,
    loop: Optional[asyncio.AbstractEventLoop] = None
) -> EyeTrackerAdapter:
    """
    Create an eye tracker adapter based on configuration.
    
    Args:
        config: System configuration containing eye tracker settings.
        loop: Asyncio event loop for callbacks.
        
    Returns:
        Configured EyeTrackerAdapter instance.
        
    Raises:
        ValueError: If eye tracker mode is invalid.
    """
    logger = get_logger()
    
    # Get event loop
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
    
    # Extract eye tracker configuration
    mode = config.eye_tracker.mode
    device_id = config.eye_tracker.device_id
    batch_size = config.eye_tracker.batch_size
    flush_interval_ms = config.eye_tracker.flush_interval_ms
    
    # Log factory configuration
    logger.system(
        "eye_tracker_adapter_factory",
        {
            "mode": mode,
            "device_id": device_id,
            "batch_size": batch_size,
            "flush_interval_ms": flush_interval_ms,
        },
        level="DEBUG"
    )
    
    # Create adapter based on mode
    if mode.upper() == "SIMULATED":
        adapter = SimulatedEyeTrackerAdapter(
            sampling_rate_hz=config.eye_tracker.simulated_sampling_rate_hz,
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
            loop=loop
        )
        logger.system(
            "eye_tracker_adapter_created",
            {"type": "simulated"},
            level="INFO"
        )
        
    elif mode.upper() == "TOBII":
        adapter = TobiiProEyeTrackerAdapter(
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
            loop=loop
        )
        logger.system(
            "eye_tracker_adapter_created",
            {"type": "tobii_pro"},
            level="INFO"
        )
        
    else:
        error_msg = (
            f"Invalid eye tracker mode: {mode}. "
            f"Valid modes are: SIMULATED, TOBII"
        )
        logger.system(
            "eye_tracker_adapter_invalid_mode",
            {"mode": mode},
            level="ERROR"
        )
        raise ValueError(error_msg)
    
    return adapter
