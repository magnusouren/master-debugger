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
from typing import Optional, List, Callable
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

METRIC_KEYGROUPS = {
    "pupil_diameter": {
        "load": {
            "mean": "pupil_mean",
            "slope": "pupil_slope",
            "vel": "pupil_mean_abs_vel",
            "std": "pupil_std",
            "range": "pupil_range",
        },
        "quality": {
            "valid_ratio": "pupil_valid_ratio",
            "count": "pupil_window_sample_count",
        },
    },
    "data_quality": {
        "quality": {
            # primary confidence signals
            "valid_ratio_any": "dq_valid_ratio_any",
            "valid_ratio_both": "dq_valid_ratio_both",

            # secondary / fallback
            "valid_ratio_left": "dq_valid_ratio_left",
            "valid_ratio_right": "dq_valid_ratio_right",

            # informational / diagnostic (not used directly in scoring yet)
            "sample_count": "dq_sample_count",
            "longest_invalid_run_ms": "dq_longest_invalid_run_ms",
            "gap_count_over_100ms": "dq_gap_count_over_100ms",
            "mean_dt_ms": "dq_mean_dt_ms",
            "dt_jitter_ms": "dq_dt_jitter_ms",
        }
    },
    # TODO - add more later:
    # "gaze_dispersion": {"load": {"dispersion": "gaze_dispersion"}, ...},
}


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

        if self._is_running:
            self.estimate()
    
    def estimate(self) -> Optional[UserStateEstimate]:
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

        estimate = UserStateEstimate(
            timestamp=windows[-1].window_end,
            score=result,
            contributing_features=self._extract_contributing_features(windows),
            model_version=None,
            model_type=self._model_type.value,
            metadata={
                "raw_score": raw_score,
                "feature_window_size": len(windows),
                "avg_valid_sample_ratio": self._avg_valid_ratio(windows),
            },
        )

        self._current_estimate = estimate

        # TODO - Should all estimates be broadcasted?
        # Only fire callbacks when actionable
        if confidence >= self._config.min_confidence_for_action:
            for cb in list(self._output_callbacks):
                try:
                    cb(estimate)
                except Exception:
                    pass

        return estimate
    
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
        Get recent history of scores.
        
        Args:
            n_samples: Number of recent samples to return.
            
        Returns:
            List of recent scores.
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
        Estimate user state using rule-based heuristics.

        Args:
            windows: List of recent feature windows.
        Returns:
            Computed user state score.
        """


        enabled = self._enabled(windows)

        # If pupil metrics aren't enabled, we can't do pupil-based load
        if "pupil_diameter" not in enabled:
            return 0.5

        load_keys = METRIC_KEYGROUPS["pupil_diameter"]["load"]
        mean_series = self._metric_series(windows, load_keys["mean"])
        slope_series = self._metric_series(windows, load_keys["slope"])
        vel_series = self._metric_series(windows, load_keys["vel"])
        std_series = self._metric_series(windows, load_keys["std"])
        range_series = self._metric_series(windows, load_keys["range"])

        if not (mean_series or slope_series or vel_series or std_series or range_series):
            return 0.5

        pupil_mean = sum(mean_series) / len(mean_series) if mean_series else 0.0
        pupil_slope = sum(slope_series) / len(slope_series) if slope_series else 0.0
        pupil_vel = sum(vel_series) / len(vel_series) if vel_series else 0.0
        pupil_std = sum(std_series) / len(std_series) if std_series else 0.0
        pupil_range = sum(range_series) / len(range_series) if range_series else 0.0

        mean_component  = self._ramp(pupil_mean,  lo=3.8, hi=4.6)
        slope_component = self._ramp(pupil_slope, lo=0.00, hi=0.30)
        vel_component   = self._ramp(pupil_vel,   lo=5.0,  hi=15.0)
        std_component   = self._ramp(pupil_std,   lo=0.1,  hi=0.5)
        range_component = self._ramp(pupil_range, lo=1.0,  hi=3.0)

        score = (
            0.4 * mean_component +
            0.2 * slope_component +
            0.2 * vel_component +
            0.1 * std_component +
            0.1 * range_component
        )

        return float(max(0.0, min(1.0, score)))
    
    def _estimate_ml_classifier(
        self, features: List[WindowFeatures]
    ) -> Optional[float]:
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
    ) -> Optional[float]:
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
        """
        Compute confidence in the current estimate.
        
        Args:
            windows: Recent feature windows.
            score: Current user state score.

        Returns:
            Confidence value between 0.0 and 1.0.
        """
        enabled = self._enabled(windows)

        # TODO - add more factors later
        ratio_keys = []
        if "data_quality" in enabled:
            dq = METRIC_KEYGROUPS["data_quality"]["quality"]
            ratio_keys.extend([
                dq["valid_ratio_any"],
                dq["valid_ratio_both"],
                dq["valid_ratio_left"],
                dq["valid_ratio_right"],
        ])
        if "pupil_diameter" in enabled:
            ratio_keys.extend([
                METRIC_KEYGROUPS["pupil_diameter"]["quality"]["valid_ratio"],
            ])

        # collect data quality ratios
        ratios = []
        for wf in windows:
            v = None
            if wf.features:
                for k in ratio_keys:
                    if k in wf.features and wf.features[k] is not None:
                        v = wf.features[k]
                        break
            if v is None:
                v = wf.valid_sample_ratio
            if v is None:
                continue
            try:
                ratios.append(float(v))
            except (TypeError, ValueError):
                continue

        dq = sum(ratios) / len(ratios) if ratios else 0.0
        qty = self._ramp(float(len(windows)), lo=3.0, hi=12.0)

        recent = [v for v in list(self._score_history)[-10:] if v is not None]
        if len(recent) >= 3:
            mean = sum(recent) / len(recent)
            var = sum((x - mean) ** 2 for x in recent) / (len(recent) - 1)
            std = math.sqrt(var)
            stability = 1.0 - self._ramp(std, lo=0.02, hi=0.15)
        else:
            stability = 0.5

        conf = 0.5 * dq + 0.3 * qty + 0.2 * stability
        return float(max(0.0, min(1.0, conf)))
    
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
        contribs = {}

        enabled = self._enabled(features)
        if "pupil_diameter" in enabled:
            load_keys = METRIC_KEYGROUPS["pupil_diameter"]["load"]
            mean_series = self._metric_series(features, load_keys["mean"])
            slope_series = self._metric_series(features, load_keys["slope"])
            vel_series = self._metric_series(features, load_keys["vel"])
            std_series = self._metric_series(features, load_keys["std"])
            range_series = self._metric_series(features, load_keys["range"])

            if mean_series:
                contribs["pupil_mean"] = sum(mean_series) / len(mean_series)
            if slope_series:
                contribs["pupil_slope"] = sum(slope_series) / len(slope_series)
            if vel_series:
                contribs["pupil_mean_abs_vel"] = sum(vel_series) / len(vel_series)
            if std_series:
                contribs["pupil_std"] = sum(std_series) / len(std_series)
            if range_series:
                contribs["pupil_range"] = sum(range_series) / len(range_series)
        
        return contribs

    # ----------------------------
    # Utilities - AI GENERATED CODE
    # ----------------------------


    def _enabled(self, windows: List[WindowFeatures]) -> List[str]:
        """
        Get list of enabled metrics from the latest window.
        Args:
            windows: List of recent feature windows.
        
        Returns:
            List of enabled metric names.
        """
        enabled = windows[-1].enabled_metrics if windows and windows[-1].enabled_metrics else []
        return enabled

    def _metric_series(self, windows: List[WindowFeatures], key: str) -> List[float]:
        """
        Extract time series for a specific metric key from windows.

        Args:
            windows: List of recent feature windows.
            key: Metric key to extract.
        
        Returns:
            List of metric values.
        """

        out: List[float] = []
        for wf in windows:
            if not wf.features or key not in wf.features:
                continue
            v = wf.features[key]
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            out.append(fv)
        return out

    def _ramp(self, x: float, lo: float, hi: float) -> float:
        """
        Linear ramp function from lo to hi.
        Returns 0.0 if x <= lo, 1.0 if x >= hi, linear in between.
        """
        if hi <= lo:
            return 1.0 if x >= hi else 0.0
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        return float((x - lo) / (hi - lo))


    def _avg_valid_ratio(self, windows: List["WindowFeatures"]) -> float:
        vals = []

        for wf in windows:
            feats = getattr(wf, "features", None) or {}

            # Get sample count for weighting
            n = feats.get("dq_sample_count", 0) or 0
            if n <= 0:
                continue

            # Choose which ratio you consider "valid" in the system
            ratio = feats.get("dq_valid_ratio_any", None)

            # Some metrics can be None → skip
            if ratio is None:
                continue

            # clamp for safety (can be useful for numerical edge cases)
            try:
                r = float(ratio)
            except (TypeError, ValueError):
                continue

            if r < 0.0:
                r = 0.0
            elif r > 1.0:
                r = 1.0

            vals.append(r)

        if not vals:
            return 0.0

        return float(sum(vals) / len(vals))