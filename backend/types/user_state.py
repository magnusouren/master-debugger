"""
Type definitions for user state estimation.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


@dataclass
class MetricBaseline:
    """Baseline statistics for a single metric."""
    mean: float
    std: float
    min_value: float
    max_value: float
    sample_count: int
    # Optional empirical bounds from baseline samples (e.g., outer 5% => 2.5/97.5 percentiles).
    p02_5: Optional[float] = None
    p97_5: Optional[float] = None


@dataclass
class ParticipantBaseline:
    """
    Baseline calibration data for a participant.
    Recorded during baseline phase (e.g., 1 minute of simple reading).
    """
    participant_id: str
    recorded_at: float  # Unix timestamp
    duration_seconds: float

    # Baseline stats for each metric
    metrics: Dict[str, MetricBaseline] = field(default_factory=dict)

    # Metadata
    task_description: str = "baseline_reading"
    is_valid: bool = True

    def get_zscore(self, metric_name: str, value: float) -> Optional[float]:
        """
        Compute z-score for a value relative to baseline.
        """
        if metric_name not in self.metrics:
            return None

        baseline = self.metrics[metric_name]
        if baseline.std <= 0:
            return 0.0

        return (value - baseline.mean) / baseline.std

    def _ramp(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0 if x >= hi else 0.0
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        return (x - lo) / (hi - lo)

    def get_normalized_score(self, metric_name: str, value: float) -> Optional[float]:
        if metric_name not in self.metrics:
            return None

        baseline = self.metrics[metric_name]

        if baseline.p02_5 is not None and baseline.p97_5 is not None:
            lo = float(baseline.p02_5)
            hi = float(baseline.p97_5)

            # Prevent overly narrow calibration ranges
            min_range_by_metric = {
                "ipa": 0.15,
                "fixation_duration_ms": 100.0,
                "anticipation_velocity": 1.0,
                "perceived_difficulty_std": 1.0,
                "ipi": 0.5,
                "cognitive_load_score": 0.2,
            }
            min_range = min_range_by_metric.get(metric_name, 0.1)

            current_range = hi - lo
            if current_range < min_range:
                center = (lo + hi) / 2.0
                lo = center - (min_range / 2.0)
                hi = center + (min_range / 2.0)

            return self._ramp(value, lo, hi)

        z = self.get_zscore(metric_name, value)
        if z is None:
            return None

        return self._ramp(z, -2.0, 2.0)


class UserStateType(Enum):
    """Types of user states that can be estimated."""
    STRESS = "stress"
    COGNITIVE_LOAD = "cognitive_load"
    CONFUSION = "confusion"
    FATIGUE = "fatigue"
    ENGAGEMENT = "engagement"
    # Add more state types as needed


@dataclass
class UserStateScore:
    """
    Scalar score representing user's current interaction state.
    Output of Reactive Tool.
    """
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    state_type: UserStateType = UserStateType.COGNITIVE_LOAD
    
    def __post_init__(self):
        """Validate score and confidence are within bounds."""
        pass  # TODO: Add validation logic


@dataclass
class UserStateEstimate:
    """
    Complete user state estimation with metadata.
    """
    timestamp: float  # When the estimate was made
    score: UserStateScore
    
    # Features used for this estimate (for interpretability)
    contributing_features: Dict[str, float] = field(default_factory=dict)
    
    # Model information
    model_version: Optional[str] = None
    model_type: Optional[str] = None  # "rule_based", "ml_classifier", "sequence_model"
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Provenance
    source_window_id: Optional[str] = None
    forecast_id: Optional[str] = None
    source_type: Optional[str] = None  # observed_features | predicted_features
    estimate_id: Optional[str] = None
