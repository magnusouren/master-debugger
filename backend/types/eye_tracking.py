"""
Type definitions for eye-tracking data structures.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class GazeSample:
    """A single gaze sample from the eye tracker."""
    timestamp: float  # Unix timestamp in seconds
    left_eye_x: Optional[float] = None
    left_eye_y: Optional[float] = None
    right_eye_x: Optional[float] = None
    right_eye_y: Optional[float] = None
    left_pupil_diameter: Optional[float] = None
    right_pupil_diameter: Optional[float] = None
    left_eye_valid: bool = False
    right_eye_valid: bool = False
    # Additional raw fields from Tobii can be added here
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawGazeData:
    """Container for raw gaze data from the eye tracker."""
    samples: List[GazeSample] = field(default_factory=list)
    device_id: Optional[str] = None
    sampling_rate_hz: float = 120.0
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None


@dataclass
class WindowFeatures:
    """
    Computed features from a window of gaze samples.
    Output of Signal Processing layer, input to Reactive Tool.
    """
    window_start: float  # Unix timestamp
    window_end: float    # Unix timestamp
    
    # TODO: Define specific metrics to extract (configurable)
    # Placeholder feature dictionary - to be expanded
    features: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    sample_count: int = 0
    valid_sample_ratio: float = 0.0
    
    # Example feature placeholders (to be defined):
    # - fixation_count: int
    # - fixation_duration_mean: float
    # - saccade_count: int
    # - saccade_amplitude_mean: float
    # - pupil_diameter_mean: float
    # - blink_rate: float
    # - gaze_dispersion: float


@dataclass
class PredictedFeatures:
    """
    Predicted future features from the Forecasting Tool.
    Same structure as WindowFeatures for downstream compatibility.
    """
    prediction_timestamp: float  # When the prediction was made
    target_window_start: float   # Predicted window start time
    target_window_end: float     # Predicted window end time
    horizon_seconds: float       # How far into the future
    
    # Predicted features (same format as WindowFeatures)
    features: Dict[str, float] = field(default_factory=dict)
    
    # Prediction confidence/uncertainty
    confidence: float = 0.0
    uncertainty: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def to_window_features(predicted: 'PredictedFeatures') -> WindowFeatures:
        """Convert PredictedFeatures to WindowFeatures format."""
        return WindowFeatures(
            window_start=predicted.target_window_start,
            window_end=predicted.target_window_end,
            features=predicted.features,
            sample_count=0,  # Not applicable for predictions
            valid_sample_ratio=0  # Not applicable for predictions
        )
