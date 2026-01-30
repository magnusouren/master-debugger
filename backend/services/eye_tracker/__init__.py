"""
Eye Tracker Adapter Service

Provides adapter interfaces and implementations for eye tracker hardware.
"""
from backend.services.eye_tracker.base import EyeTrackerAdapter, AdapterState
from backend.services.eye_tracker.simulated_adapter import SimulatedEyeTrackerAdapter
from backend.services.eye_tracker.tobii_pro_adapter import TobiiProEyeTrackerAdapter
from backend.services.eye_tracker.factory import create_eye_tracker_adapter

__all__ = [
    "EyeTrackerAdapter",
    "AdapterState",
    "SimulatedEyeTrackerAdapter",
    "TobiiProEyeTrackerAdapter",
    "create_eye_tracker_adapter",
]
