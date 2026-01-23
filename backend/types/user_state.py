"""
Type definitions for user state estimation.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


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
