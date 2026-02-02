"""
Reactive Tool

Input: Sliding window of features
Output: user_state_score (0–1) + confidence

This layer continuously estimates a scalar user_state_score representing 
the user's current interaction state (e.g., stress, load, or related 
behavioral effects) based on recent feature windows.

Model progression:
- Baseline: rule-based thresholds
- Next: classical ML models (e.g., logistic regression, random forest)
- Later: sequence-based models
"""
import math
from typing import Optional, List, Callable, Tuple, Union
from collections import deque
from enum import Enum

from backend.types import (
    WindowFeatures,
    UserStateScore,
    UserStateEstimate,
    ReactiveToolConfig,
)
from backend.types.user_state import UserStateType


class ModelType(Enum):
    """Available model types for state estimation."""
    RULE_BASED = "rule_based"
    ML_CLASSIFIER = "ml_classifier"
    SEQUENCE_MODEL = "sequence_model"


class ReactiveTool:
    """
    Estimates user state from eye-tracking features.
    """
    
    def __init__(self, config: Optional[ReactiveToolConfig] = None):
        """
        Initialize the Reactive Tool.
        
        Args:
            config: Configuration for reactive tool parameters.
        """
        self._config = config or ReactiveToolConfig()
        self._feature_window: deque[WindowFeatures] = deque()
        self._current_estimate: Optional[UserStateEstimate] = None
        self._model: Optional[object] = None  # TODO: Define model interface
        self._model_type: ModelType = ModelType.RULE_BASED
        self._output_callbacks: List[Callable[[UserStateEstimate], None]] = []
        self._score_history: deque[float] = deque()  # For smoothing
        self._is_running: bool = False
    
    def configure(self, config: ReactiveToolConfig) -> None:
        """
        Update reactive tool configuration.
        
        Args:
            config: New configuration to apply.
        """
        self._config = config
    
    def start(self) -> None:
        """Start state estimation."""
        self._is_running = True
    
    def stop(self) -> None:
        """Stop state estimation."""
        self._is_running = False
    
    def reset(self) -> None:
        """Reset internal state and sliding window."""
        self._feature_window.clear()
        self._current_estimate = None
        self._score_history.clear()
        self._is_running = False
    
    def set_model_type(self, model_type: ModelType) -> None:
        """
        Set the type of model to use for estimation.
        
        Args:
            model_type: The model type to use.
        """
        self._model_type = model_type
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained ML model for state estimation.
        
        Args:
            model_path: Path to the model file.
            
        Returns:
            True if model loaded successfully.
        """
        pass  # TODO: Implement model loading
    
    def add_features(self, features: WindowFeatures) -> None:
        """
        Add new window-based features and keep only features
        within the last window_size_seconds (e.g. last 60s).
        """
        self._feature_window.append(features)

        # Time-based sliding window
        horizon = self._config.window_size_seconds  # e.g. 60 seconds
        cutoff_ts = features.window_end - horizon

        # Remove old windows
        while self._feature_window and self._feature_window[0].window_end < cutoff_ts:
            self._feature_window.popleft()
            
        if not self._is_running:
            return None

        return self.estimate()
    
    def estimate(self) -> Optional[UserStateScore]:
        """
        Compute current user state score from the sliding window.
        """
        if not self._is_running:
            return None

        windows = list(self._feature_window)
        if len(windows) < 3:
            return None
    

        # --- compute raw score ---
        if self._model_type == ModelType.RULE_BASED:
            raw_score = self._estimate_rule_based(windows)
        elif self._model_type == ModelType.ML_CLASSIFIER:
            raw_score = self._estimate_ml_classifier(windows)
        else:
            raw_score = self._estimate_sequence_model(windows)

        if raw_score is None:
            raw_score = 0.5  # neutral fallback

        raw_score = float(raw_score)
        raw_score = max(0.0, min(1.0, raw_score))
        score = self._smooth_score(raw_score)

        confidence = self._compute_confidence(windows, score)

        result = UserStateScore(
            score=score,
            confidence=confidence,
            state_type=UserStateType.COGNITIVE_LOAD,
        )

        self._score_history.append(score)

        # Drop low-confidence estimates
        if confidence < self._config.min_confidence_for_action:
            return None

        self._current_estimate = result

        for cb in list(self._output_callbacks):
            try:
                cb(result)
            except Exception:
                pass

        return result
    
    def get_current_score(self) -> Optional[UserStateScore]:
        """
        Get the current user state score.
        
        Returns:
            Current score or None if not available.
        """
        if self._current_estimate:
            return self._current_estimate.score
        return None
    
    def get_score_history(self, n_samples: int = 10) -> List[float]:
        """
        Get recent history of state scores.
        
        Args:
            n_samples: Number of recent samples to return.
            
        Returns:
            List of recent score values.
        """
        return list(self._score_history)[-n_samples:]
    
    def register_output_callback(
        self, callback: Callable[[UserStateEstimate], None]
    ) -> None:
        """
        Register a callback to receive state estimates.
        
        Args:
            callback: Function to call with estimates.
        """
        if callback not in self._output_callbacks:
            self._output_callbacks.append(callback)
    
    def unregister_output_callback(
        self, callback: Callable[[UserStateEstimate], None]
    ) -> None:
        """
        Unregister a previously registered callback.
        
        Args:
            callback: The callback function to remove.
        """
        if callback in self._output_callbacks:
            self._output_callbacks.remove(callback)
    
    def update_thresholds(self, thresholds: dict) -> None:
        """
        Update rule-based thresholds.
        
        Args:
            thresholds: Dictionary of threshold values.
        """
        self._config.thresholds.update(thresholds)
    
    # --- Internal methods ---
    
    def _estimate_rule_based(self, windows: List[WindowFeatures]) -> float:
        """
        Rule-based scalar score in [0,1] using available pupil-based signals.
        """

        # Use real load-ish metrics
        pupil_mean_series = self._metric_series(windows, "pupil_mean")
        pupil_slope_series = self._metric_series(windows, "pupil_slope")
        pupil_vel_series = self._metric_series(windows, "pupil_mean_abs_vel")

        # If you have no usable metrics, return neutral
        if not (pupil_mean_series or pupil_slope_series or pupil_vel_series):
            return 0.5

        # Summaries
        pupil_mean = sum(pupil_mean_series) / len(pupil_mean_series) if pupil_mean_series else 0.0
        pupil_slope = sum(pupil_slope_series) / len(pupil_slope_series) if pupil_slope_series else 0.0
        pupil_vel = sum(pupil_vel_series) / len(pupil_vel_series) if pupil_vel_series else 0.0

        # Ramps (placeholders! calibrate later)
        # pupil_mean: your example is ~4.29
        mean_component  = self._ramp(pupil_mean,  lo=3.8, hi=4.6)

        # pupil_slope: your example is ~0.22 (which is huge compared to my earlier placeholder)
        # so set a broader ramp for now:
        slope_component = self._ramp(pupil_slope, lo=0.00, hi=0.30)

        # mean abs velocity: your example ~9.85
        # scale ramp around typical range you see:
        vel_component   = self._ramp(pupil_vel,   lo=5.0,  hi=15.0)

        # Combine into one score
        score = (
            0.55 * mean_component +
            0.25 * slope_component +
            0.20 * vel_component
        )

        return float(max(0.0, min(1.0, score)))
    
    def _estimate_ml_classifier(
        self, features: List[WindowFeatures]
    ) -> UserStateScore:
        """
        Estimate user state using ML classifier.
        
        Args:
            features: Feature window for estimation.
            
        Returns:
            Computed user state score.
        """
        pass  # TODO: Implement ML-based estimation
    
    def _estimate_sequence_model(
        self, features: List[WindowFeatures]
    ) -> UserStateScore:
        """
        Estimate user state using sequence model.
        
        Args:
            features: Feature window for estimation.
            
        Returns:
            Computed user state score.
        """
        pass  # TODO: Implement sequence model estimation
    
    def _smooth_score(self, raw_score: float) -> float:
        """
        Apply exponential moving average smoothing to score.
        
        Args:
            raw_score: Unsmoothed score value.
            
        Returns:
            Smoothed score value.
        """
        alpha = float(self._config.score_smoothing_factor)

        if raw_score is None:
            raw_score = 0.5

        raw = float(raw_score)

        # Find last valid value in history (skip None)
        prev = None
        for v in reversed(self._score_history):
            if v is not None:
                prev = float(v)
                break

        if prev is None:
            smoothed = raw
        else:
            smoothed = alpha * raw + (1.0 - alpha) * prev

        self._score_history.append(smoothed)

        # bound history
        while len(self._score_history) > 60:
            self._score_history.popleft()

        return float(max(0.0, min(1.0, smoothed)))
        
    def _compute_confidence(self, windows: List[WindowFeatures], score: float) -> float:
        valid_ratios = [wf.valid_sample_ratio for wf in windows if wf.sample_count > 0]
        dq = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0

        qty = self._ramp(len(windows), lo=3, hi=12)
    
        recent = list(self._score_history)[-10:]
        if len(recent) >= 3:
            mean = sum(recent) / len(recent)
            var = sum((x - mean) ** 2 for x in recent) / (len(recent) - 1)
            std = math.sqrt(var)
            stability = 1.0 - self._ramp(std, lo=0.02, hi=0.15)
        else:
            stability = 0.5

        conf = 0.5 * dq + 0.3 * qty + 0.2 * stability
        return max(0.0, min(1.0, conf))
    
    def _extract_contributing_features(
        self, features: List[WindowFeatures]
    ) -> dict:
        """
        Identify features that contributed most to the estimate.
        
        Args:
            features: Feature window used for estimation.
            
        Returns:
            Dictionary of feature contributions.
        """
        pass  # TODO: Implement feature contribution extraction


    # ----------------------------
    # Utilities - AI GENERATED CODE
    # ----------------------------

    def _metric_series(self, windows: List[WindowFeatures], key: str) -> List[float]:
        out: List[float] = []
        for wf in windows:
            if not wf.features:
                continue
            if key not in wf.features:
                continue
            try:
                out.append(float(wf.features[key]))
            except (TypeError, ValueError):
                continue
        return out


    def _mean_and_slope(self, series: List[float]) -> Tuple[float, float]:
        if not series:
            return 0.0, 0.0
        mean = sum(series) / len(series)
        if len(series) < 2:
            return mean, 0.0
        slope = (series[-1] - series[0]) / (len(series) - 1)
        return float(mean), float(slope)


    def _ramp(self, x: float, lo: float, hi: float) -> float:

        if x is None or lo is None or hi is None:
        # swap for your logger if you have it
            print(f"[RAMP_NONE] x={x} lo={lo} hi={hi}")
            return 0.0


        if hi <= lo:
            return 1.0 if x >= hi else 0.0
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        return float((x - lo) / (hi - lo))


    def _avg_valid_ratio(self, windows: List[WindowFeatures]) -> float:
        vals = [wf.valid_sample_ratio for wf in windows if wf.sample_count > 0]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))