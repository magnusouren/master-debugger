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
import time
from typing import Optional, List, Callable, Dict
from collections import deque
from enum import Enum

from backend.services.logger_service import LoggerService
from backend.types import (
    WindowFeatures,
    UserStateScore,
    UserStateEstimate,
    ReactiveToolConfig,
    MetricBaseline,
    ParticipantBaseline,
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
            "ipa": "pupil_ipa",  # Index of Pupillary Activity
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
    "fixation_duration": {
        "load": {
            "mean_duration_ms": "fixation_mean_duration_ms",
            "max_duration_ms": "fixation_max_duration_ms",
            "count": "fixation_count",
            "mean_dispersion": "fixation_mean_dispersion",
        },
    },
    "saccade_amplitude": {
        "load": {
            "mean_amplitude": "saccade_mean_amplitude",
            "max_amplitude": "saccade_max_amplitude",
            "count": "saccade_count",
            "mean_duration_ms": "saccade_mean_duration_ms",
            "mean_velocity": "saccade_mean_velocity",  # For anticipation (NOT cognitive load)
            "velocity_std": "saccade_velocity_std",    # For perceived difficulty
        },
    },
    "ipi": {
        "load": {
            "value": "ipi_value",  # Information Processing Index from crunchwiz
            "ratio_count": "ipi_ratio_count",
            "calibrated": "ipi_calibrated",
        },
    },
}


class ReactiveTool:
    """
    Estimates user state from eye-tracking features.
    """
    BASELINE_SCORE_METRIC = "cognitive_load_score"
    
    def __init__(
        self,
        config: Optional[ReactiveToolConfig] = None,
        logger: Optional[LoggerService] = None,
    ):
        """
        Initialize the Reactive Tool.
        
        Args:
            config: Configuration for reactive tool parameters.
            logger: Logger instance for recording events.
        """
        self._config = config or ReactiveToolConfig()
        self._feature_window: deque[WindowFeatures] = deque()
        self._current_estimate: Optional[UserStateEstimate] = None
        self._model: Optional[object] = None  # TODO: Define model interface
        self._model_type: ModelType = ModelType.RULE_BASED
        self._output_callbacks: List[Callable[[UserStateEstimate], None]] = []
        self._score_history: deque[float] = deque()  # For smoothing
        self._is_running: bool = False

        # Baseline calibration
        self._baseline: Optional[ParticipantBaseline] = None
        self._is_recording_baseline: bool = False
        self._baseline_start_time: float = 0.0
        self._baseline_samples: Dict[str, List[float]] = {}  # metric_name -> values
        self._baseline_feature_windows: List[WindowFeatures] = []
        self._feedback_trigger_bounds: Optional[Dict[str, float | int | str]] = None

        # Use provided logger or create fallback
        if logger is None:
            from backend.services.logger_service import get_logger
            logger = get_logger()
        self._logger = logger
    
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
        """Reset internal state, sliding window, and baseline."""
        self._feature_window.clear()
        self._current_estimate = None
        self._score_history.clear()
        self._is_running = False
        # Clear baseline calibration
        self._baseline = None
        self._is_recording_baseline = False
        self._baseline_samples = {}
        self._baseline_feature_windows = []
        self._feedback_trigger_bounds = None

    # --- Baseline calibration methods ---

    def start_baseline_recording(self, participant_id: str) -> None:
        """
        Start recording baseline metrics for a participant.

        Call this when the participant begins the baseline task (e.g., reading simple text).
        """
        self._is_recording_baseline = True
        self._baseline_start_time = time.time()
        self._baseline_feature_windows = []
        self._feedback_trigger_bounds = None
        self._baseline_samples = {
            "ipa": [],
            "fixation_duration_ms": [],
            "anticipation_velocity": [],
            "perceived_difficulty_std": [],
            "ipi": [],  # Information Processing Index from crunchwiz
            self.BASELINE_SCORE_METRIC: [],  # Final score used for delivery triggering
        }
        self._logger.system(
            "baseline_recording_started",
            {"participant_id": participant_id},
            level="INFO"
        )

    def stop_baseline_recording(self, participant_id: str) -> Optional[ParticipantBaseline]:
        """
        Stop recording and compute baseline statistics.

        Call this when the baseline task is complete.

        Returns:
            ParticipantBaseline object with computed statistics, or None if insufficient data.
        """
        if not self._is_recording_baseline:
            return None

        self._is_recording_baseline = False
        duration = time.time() - self._baseline_start_time

        # Compute statistics for each metric
        metrics: Dict[str, MetricBaseline] = {}
        for metric_name, values in self._baseline_samples.items():
            if len(values) >= 3:  # Need at least 3 samples
                mean_val = sum(values) / len(values)
                std_val = math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))
                p02_5 = self._percentile(values, 2.5)
                p97_5 = self._percentile(values, 97.5)
                metrics[metric_name] = MetricBaseline(
                    mean=mean_val,
                    std=max(std_val, 0.001),  # Avoid zero std
                    min_value=min(values),
                    max_value=max(values),
                    sample_count=len(values),
                    p02_5=p02_5,
                    p97_5=p97_5,
                )

        if not metrics:
            self._logger.system(
                "baseline_recording_failed",
                {"participant_id": participant_id, "reason": "insufficient_data"},
                level="ERROR"
            )
            return None

        self._baseline = ParticipantBaseline(
            participant_id=participant_id,
            recorded_at=self._baseline_start_time,
            duration_seconds=duration,
            metrics=metrics,
            is_valid=True,
        )

        self._logger.system(
            "baseline_recording_completed",
            {
                "participant_id": participant_id,
                "duration_seconds": round(duration, 1),
                "metrics": {
                    k: {
                        "sample_count": v.sample_count,
                        "mean": round(v.mean, 4),
                        "std": round(v.std, 4),
                        "min": round(v.min_value, 4),
                        "max": round(v.max_value, 4),
                        "range": round(v.max_value - v.min_value, 4),
                        "p02_5": round(v.p02_5, 4) if v.p02_5 is not None else None,
                        "p97_5": round(v.p97_5, 4) if v.p97_5 is not None else None,
                        "percentile_range": round(v.p97_5 - v.p02_5, 4)
                        if v.p02_5 is not None and v.p97_5 is not None else None,
                    }
                    for k, v in metrics.items()
                },
            },
            level="INFO"
        )

        return self._baseline

    def set_baseline(self, baseline: ParticipantBaseline) -> None:
        """Set a pre-recorded baseline for normalization."""
        self._baseline = baseline
        self._logger.system(
            "baseline_loaded",
            {"participant_id": baseline.participant_id},
            level="INFO"
        )

    def clear_baseline(self) -> None:
        """Clear the current baseline (revert to static thresholds)."""
        self._baseline = None
        self._feedback_trigger_bounds = None
        self._logger.system("baseline_cleared", {}, level="INFO")

    def has_baseline(self) -> bool:
        """Check if a valid baseline is loaded."""
        return self._baseline is not None and self._baseline.is_valid

    def get_feedback_trigger_bounds(self, std_multiplier: float = 2.0) -> Optional[Dict[str, float | int | str]]:
        """
        Return persisted trigger bounds calibrated after baseline activation.

        Parameter kept for backward compatibility with existing call sites.
        """
        _ = std_multiplier
        if self._feedback_trigger_bounds is None:
            return None
        return dict(self._feedback_trigger_bounds)

    def calibrate_feedback_trigger_bounds_from_baseline_windows(
        self,
    ) -> Optional[Dict[str, float | int | str]]:
        """Calibrate persistent trigger bounds after baseline is active.

        Two-step calibration is required: we first compute ParticipantBaseline,
        then rescore the recorded baseline windows in that calibrated space.
        """
        self._feedback_trigger_bounds = None

        if self._baseline is None or not self._baseline.is_valid:
            self._logger.system(
                "feedback_trigger_bounds_calibration_fallback",
                {"reason": "baseline_missing_or_invalid"},
                level="INFO",
            )
            return None

        observed_windows = [
            wf
            for wf in self._baseline_feature_windows
            if not getattr(wf, "is_predicted", False)
        ]
        if len(observed_windows) < 3:
            self._logger.system(
                "feedback_trigger_bounds_calibration_fallback",
                {
                    "reason": "insufficient_observed_baseline_windows",
                    "window_count": len(observed_windows),
                },
                level="INFO",
            )
            return None

        scores = self._score_windows_in_current_calibrated_space(observed_windows)
        if len(scores) < 3:
            self._logger.system(
                "feedback_trigger_bounds_calibration_fallback",
                {
                    "reason": "insufficient_calibrated_scores",
                    "score_count": len(scores),
                },
                level="INFO",
            )
            return None

        mean_val = sum(scores) / len(scores)
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in scores) / len(scores))
        std_val = max(std_val, 0.001)
        p02_5 = self._percentile(scores, 2.5)
        p97_5 = self._percentile(scores, 97.5)

        if p02_5 is not None and p97_5 is not None:
            lower = float(p02_5)
            upper = float(p97_5)
            rule = "baseline_empirical_p2_5_p97_5"
        else:
            lower = mean_val - (2.0 * std_val)
            upper = mean_val + (2.0 * std_val)
            rule = "baseline_mean_pm_2sd"

        self._feedback_trigger_bounds = {
            "rule": rule,
            "mean": float(mean_val),
            "std": float(std_val),
            "lower": float(lower),
            "upper": float(upper),
            "sample_count": int(len(scores)),
        }

        self._baseline.metrics[self.BASELINE_SCORE_METRIC] = MetricBaseline(
            mean=float(mean_val),
            std=float(std_val),
            min_value=float(min(scores)),
            max_value=float(max(scores)),
            sample_count=int(len(scores)),
            p02_5=p02_5,
            p97_5=p97_5,
        )

        trigger_width = float(upper - lower)
        score_min = float(min(scores))
        score_max = float(max(scores))
        score_range = float(score_max - score_min)

        self._logger.system(
            "feedback_trigger_bounds_calibrated",
            {
                **dict(self._feedback_trigger_bounds),
                "trigger_width": trigger_width,
                "score_min": score_min,
                "score_max": score_max,
                "score_range": score_range,
                "score_p02_5": float(p02_5) if p02_5 is not None else None,
                "score_p97_5": float(p97_5) if p97_5 is not None else None,
            },
            level="INFO",
        )
        
        return dict(self._feedback_trigger_bounds)

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> Optional[float]:
        """
        Compute percentile using linear interpolation between closest ranks.
        """
        if not values:
            return None

        sorted_vals = sorted(float(v) for v in values)
        if len(sorted_vals) == 1:
            return sorted_vals[0]

        p = max(0.0, min(100.0, float(percentile)))
        rank = (len(sorted_vals) - 1) * (p / 100.0)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return sorted_vals[lo]
        frac = rank - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    def is_recording_baseline(self) -> bool:
        """Check if baseline recording is currently active."""
        return self._is_recording_baseline

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
        if self._is_recording_baseline and not getattr(features, "is_predicted", False):
            # Keep observed baseline windows so we can rescore them after baseline is set.
            self._baseline_feature_windows.append(features)

        self._feature_window.append(features)

        # Time-based sliding window
        horizon = self._config.window_size_seconds  # e.g. 60 seconds
        cutoff_ts = features.window_end - horizon

        # Remove old windows
        while self._feature_window and self._feature_window[0].window_end < cutoff_ts:
            self._feature_window.popleft()

        if self._is_running:
            self.estimate()

    def _score_windows_in_current_calibrated_space(self, windows: List[WindowFeatures]) -> List[float]:
        """Replay observed windows through the current calibrated scoring pipeline."""
        if not windows:
            return []

        ordered = sorted(windows, key=lambda w: w.window_end)
        horizon = self._config.window_size_seconds
        alpha = float(self._config.score_smoothing_factor)
        local_windows: deque[WindowFeatures] = deque()
        local_history: deque[float] = deque()
        scores: List[float] = []

        for wf in ordered:
            local_windows.append(wf)
            cutoff_ts = wf.window_end - horizon
            while local_windows and local_windows[0].window_end < cutoff_ts:
                local_windows.popleft()

            active_windows = list(local_windows)
            if len(active_windows) < 3:
                continue

            if self._model_type == ModelType.RULE_BASED:
                raw_score = self._estimate_rule_based(active_windows)
            elif self._model_type == ModelType.ML_CLASSIFIER:
                raw_score = self._estimate_ml_classifier(active_windows)
            else:
                raw_score = self._estimate_sequence_model(active_windows)

            if raw_score is None:
                raw_score = 0.5

            raw = max(0.0, min(1.0, float(raw_score)))
            prev = local_history[-1] if local_history else None
            smoothed = raw if prev is None else (alpha * raw + (1.0 - alpha) * prev)

            local_history.append(smoothed)
            while len(local_history) > 60:
                local_history.popleft()

            scores.append(float(smoothed))

        return scores
    
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

        # Keep a baseline distribution of the final score used by delivery trigger logic.
        if self._is_recording_baseline and self.BASELINE_SCORE_METRIC in self._baseline_samples:
            self._baseline_samples[self.BASELINE_SCORE_METRIC].append(score)

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
                "using_baseline": self.has_baseline(),
                "window_ids": [w.window_id for w in windows if getattr(w, "window_id", None)],
                "source_window_id": windows[-1].window_id,
                "forecast_id": getattr(windows[-1], "forecast_id", None),
                "source_type": "predicted_features" if windows[-1].is_predicted else "observed_features",
            },
            source_window_id=windows[-1].window_id,
            forecast_id=getattr(windows[-1], "forecast_id", None),
            source_type="predicted_features" if windows[-1].is_predicted else "observed_features",
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

    def get_latest_estimate(self) -> Optional[UserStateEstimate]:
        """Return the latest computed estimate without side effects."""
        return self._current_estimate
    
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
        Estimate user state using rule-based heuristics with strict 5 equal-weight metrics.

        Uses baseline-relative normalization if a baseline is available,
        otherwise falls back to static thresholds.

        Metrics (each weighted 0.2):
        1. IPA (Index of Pupillary Activity) - cognitive load from pupil dynamics
        2. Fixation Duration - longer fixations indicate processing difficulty
        3. Anticipation (saccade velocity) - NOT cognitive load, measures anticipation/preparation
        4. Perceived Difficulty (saccade velocity std) - variable scanning indicates uncertainty
        5. IPI (Information Processing Index) - ratio of short vs long fixation-saccade patterns
           Lower IPI = deeper processing = higher cognitive load

        Args:
            windows: List of recent feature windows.
        Returns:
            Computed user state score (0.0 to 1.0).
        """
        enabled = self._enabled(windows)

        # components: metric_name -> sub_score
        # All metrics contribute exactly 0.2 each. Missing values fallback to neutral 0.5.
        components: dict[str, float] = {}
        raw_values: dict = {}
        missing_metrics: list[str] = []

        # --- 1. IPA - Index of Pupillary Activity (weight 0.2) ---
        if "pupil_diameter" in enabled:
            ipa_series = self._metric_series(
                windows, METRIC_KEYGROUPS["pupil_diameter"]["load"]["ipa"]
            )
            if ipa_series:
                mean_ipa = sum(ipa_series) / len(ipa_series)
                raw_values["ipa"] = mean_ipa
                score = self._normalize_metric("ipa", mean_ipa, fallback_lo=0.5, fallback_hi=2.5)
                components["ipa"] = score

        # --- 2. Fixation Duration (weight 0.2) ---
        if "fixation_duration" in enabled:
            dur_series = self._metric_series(
                windows, METRIC_KEYGROUPS["fixation_duration"]["load"]["mean_duration_ms"]
            )
            if dur_series:
                mean_dur = sum(dur_series) / len(dur_series)
                raw_values["fixation_duration_ms"] = mean_dur
                score = self._normalize_metric("fixation_duration_ms", mean_dur, fallback_lo=150.0, fallback_hi=500.0)
                components["fixation_duration"] = score

        # --- 3. Anticipation - Saccade Velocity (weight 0.2) ---
        if "saccade_amplitude" in enabled:
            vel_series = self._metric_series(
                windows, METRIC_KEYGROUPS["saccade_amplitude"]["load"]["mean_velocity"]
            )
            if vel_series:
                mean_vel = sum(vel_series) / len(vel_series)
                raw_values["anticipation_velocity"] = mean_vel
                score = self._normalize_metric("anticipation_velocity", mean_vel, fallback_lo=1.0, fallback_hi=5.0)
                components["anticipation"] = score

        # --- 4. Perceived Difficulty - Saccade Velocity Variability (weight 0.2) ---
        # Compute std of mean velocities across windows (aggregated approach)
        if "saccade_amplitude" in enabled:
            vel_series = self._metric_series(
                windows, METRIC_KEYGROUPS["saccade_amplitude"]["load"]["mean_velocity"]
            )
            if len(vel_series) >= 2:
                # Compute std of velocities across windows
                mean_vel = sum(vel_series) / len(vel_series)
                variance = sum((v - mean_vel) ** 2 for v in vel_series) / len(vel_series)
                velocity_std = math.sqrt(variance)
                raw_values["perceived_difficulty_std"] = velocity_std
                score = self._normalize_metric("perceived_difficulty_std", velocity_std, fallback_lo=0.5, fallback_hi=10.0)
                components["perceived_difficulty"] = score

        # --- 5. Information Processing Index (weight 0.2) ---
        # Uses IPI from signal processing (crunchwiz formula)
        # IPI = count(short_fix_short_sac) / count(long_fix_short_sac)
        # Higher IPI = rapid scanning, Lower IPI = deeper processing
        if "ipi" in enabled:
            ipi_series = self._metric_series(
                windows, METRIC_KEYGROUPS["ipi"]["load"]["value"]
            )
            if ipi_series:
                mean_ipi = sum(ipi_series) / len(ipi_series)
                raw_values["ipi"] = mean_ipi
                # Lower IPI indicates deeper processing (higher cognitive load)
                # So we invert: high IPI (scanning) = low load, low IPI (focused) = high load
                score = 1.0 - self._normalize_metric("ipi", mean_ipi, fallback_lo=0.5, fallback_hi=2.0)
                components["ipi"] = score

        expected_components = [
            "ipa",
            "fixation_duration",
            "anticipation",
            "perceived_difficulty",
            "ipi",
        ]

        active_component_names = list(components.keys())
        missing_metrics = [name for name in expected_components if name not in components]

        # Record samples if in baseline recording mode
        if self._is_recording_baseline:
            for metric_name, value in raw_values.items():
                if metric_name in self._baseline_samples:
                    self._baseline_samples[metric_name].append(value)

        # Dynamic equal weighting across only available components
        if not active_component_names:
            score = 0.5
        else:
            score = sum(components[name] for name in active_component_names) / len(active_component_names)

        # Log individual metrics with raw values and normalized scores
        self._logger.system(
            "user_state_metrics",
            {
                "raw_values": {k: round(v, 4) if v is not None else None for k, v in raw_values.items()},
                "normalized_scores": {k: round(s, 3) for k, s in components.items()},
                "final_score": round(score, 3),
                "active_metrics": [k for k in expected_components if k not in missing_metrics],
                "missing_metrics_fallback": missing_metrics,
                "using_baseline": self.has_baseline(),
            },
            level="DEBUG"
        )

        return float(max(0.0, min(1.0, score)))

    def _normalize_metric(
        self,
        metric_name: str,
        value: float,
        fallback_lo: float,
        fallback_hi: float
    ) -> float:
        """
        Normalize a metric value to 0-1 range.

        Uses baseline-relative normalization if available, otherwise static thresholds.

        Args:
            metric_name: Name of the metric
            value: Raw metric value
            fallback_lo: Static low threshold (if no baseline)
            fallback_hi: Static high threshold (if no baseline)

        Returns:
            Normalized score (0.0 to 1.0)
        """
        if self._baseline is not None:
            normalized = self._baseline.get_normalized_score(metric_name, value)
            if normalized is not None:
                return normalized

        # Fallback to static ramp
        return self._ramp(value, lo=fallback_lo, hi=fallback_hi)
    
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
        Extract the 5 key metrics used in scoring.

        Args:
            features: Feature window used for estimation.

        Returns:
            Dictionary with raw metric values and normalized scores.
        """
        contribs = {}
        enabled = self._enabled(features)

        # --- 1. IPA (Index of Pupillary Activity) ---
        if "pupil_diameter" in enabled:
            ipa_series = self._metric_series(
                features, METRIC_KEYGROUPS["pupil_diameter"]["load"]["ipa"]
            )
            if ipa_series:
                mean_ipa = sum(ipa_series) / len(ipa_series)
                contribs["ipa_raw"] = round(mean_ipa, 4)
                contribs["ipa_score"] = round(
                    self._normalize_metric("ipa", mean_ipa, fallback_lo=0.5, fallback_hi=2.5), 3
                )

        # --- 2. Fixation Duration ---
        if "fixation_duration" in enabled:
            dur_series = self._metric_series(
                features, METRIC_KEYGROUPS["fixation_duration"]["load"]["mean_duration_ms"]
            )
            if dur_series:
                mean_dur = sum(dur_series) / len(dur_series)
                contribs["fixation_duration_ms"] = round(mean_dur, 1)
                contribs["fixation_duration_score"] = round(
                    self._normalize_metric("fixation_duration_ms", mean_dur, fallback_lo=150.0, fallback_hi=500.0), 3
                )

        # --- 3. Anticipation (Saccade Velocity) ---
        if "saccade_amplitude" in enabled:
            vel_series = self._metric_series(
                features, METRIC_KEYGROUPS["saccade_amplitude"]["load"]["mean_velocity"]
            )
            if vel_series:
                mean_vel = sum(vel_series) / len(vel_series)
                contribs["anticipation_velocity"] = round(mean_vel, 4)
                contribs["anticipation_score"] = round(
                    self._normalize_metric("anticipation_velocity", mean_vel, fallback_lo=1.0, fallback_hi=5.0), 3
                )

        # --- 4. Perceived Difficulty (Saccade Velocity Std across windows) ---
        if "saccade_amplitude" in enabled:
            vel_series = self._metric_series(
                features, METRIC_KEYGROUPS["saccade_amplitude"]["load"]["mean_velocity"]
            )
            if len(vel_series) >= 2:
                mean_vel = sum(vel_series) / len(vel_series)
                variance = sum((v - mean_vel) ** 2 for v in vel_series) / len(vel_series)
                velocity_std = math.sqrt(variance)
                contribs["perceived_difficulty_std"] = round(velocity_std, 4)
                contribs["perceived_difficulty_score"] = round(
                    self._normalize_metric("perceived_difficulty_std", velocity_std, fallback_lo=0.5, fallback_hi=10.0), 3
                )

        # --- 5. Information Processing Index (from signal processing) ---
        if "ipi" in enabled:
            ipi_series = self._metric_series(
                features, METRIC_KEYGROUPS["ipi"]["load"]["value"]
            )
            if ipi_series:
                mean_ipi = sum(ipi_series) / len(ipi_series)
                contribs["ipi_raw"] = round(mean_ipi, 4)
                # Inverted: low IPI = high load
                contribs["ipi_score"] = round(
                    1.0 - self._normalize_metric("ipi", mean_ipi, fallback_lo=0.5, fallback_hi=2.0), 3
                )

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
