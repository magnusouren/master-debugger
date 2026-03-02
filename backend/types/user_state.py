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

        Args:
            metric_name: Name of the metric (e.g., "ipa", "fixation_duration_ms")
            value: Current metric value

        Returns:
            Z-score or None if metric not in baseline or std is 0
        """
        if metric_name not in self.metrics:
            return None
        baseline = self.metrics[metric_name]
        if baseline.std <= 0:
            return 0.0  # No variation in baseline
        return (value - baseline.mean) / baseline.std

    def get_normalized_score(self, metric_name: str, value: float) -> Optional[float]:
        """
        Convert z-score to 0-1 range using sigmoid-like mapping.

        Z-scores are mapped: -2 → ~0.12, 0 → 0.5, +2 → ~0.88

        Args:
            metric_name: Name of the metric
            value: Current metric value

        Returns:
            Normalized score (0-1) or None if metric not in baseline
        """
        zscore = self.get_zscore(metric_name, value)
        if zscore is None:
            return None
        # Sigmoid mapping: zscore of ±2 maps to ~0.12/0.88
        # zscore of ±3 maps to ~0.05/0.95
        import math
        return 1.0 / (1.0 + math.exp(-zscore))


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
