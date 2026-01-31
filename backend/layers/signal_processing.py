"""
Signal Processing Layer

Input: Raw eye-tracking data (120 Hz)
Output: Window-based features at a lower frequency (e.g., 2–10 Hz)
Configuration: Selected metrics, window length, output frequency

This layer ingests raw data from the Tobii eye tracker and computes 
window-based features suitable for downstream analysis. It handles 
missing values and invalid samples and outputs a stable, structured 
representation of the selected metrics.
"""
import math
from typing import Any, Dict, Optional, List, Callable
from collections import deque
from backend.services.logger_service import get_logger


from backend.types import (
    RawGazeData,
    GazeSample,
    WindowFeatures,
    SignalProcessingConfig,
)


class SignalProcessingLayer:
    """
    Processes raw eye-tracking data into window-based features.
    """
    
    def __init__(self, config: Optional[SignalProcessingConfig] = None):
        """
        Initialize the Signal Processing layer.
        
        Args:
            config: Configuration for signal processing parameters.
        """
        self._config = config or SignalProcessingConfig()
        self._sample_buffer: deque[GazeSample] = deque()
        self._output_callbacks: List[Callable[[WindowFeatures], None]] = []
        self._last_output_time: float = 0.0
        self._is_running: bool = False
        self._logger = get_logger()
        self._next_window_end_ts: Optional[float] = None
    
    def configure(self, config: SignalProcessingConfig) -> None:
        """
        Update layer configuration.
        
        Args:
            config: New configuration to apply.
        """
        self._config = config
        self._logger.system(
            "signal_processing_config_updated",
            {
                "input_sampling_rate_hz": config.input_sampling_rate_hz,
                "window_length_seconds": config.window_length_seconds,
                "window_overlap_ratio": config.window_overlap_ratio,
                "output_frequency_hz": config.output_frequency_hz,
                "enabled_metrics": config.enabled_metrics,
                "min_valid_sample_ratio": config.min_valid_sample_ratio,
                "interpolate_missing": config.interpolate_missing,
                "max_gap_to_interpolate_ms": config.max_gap_to_interpolate_ms,
            },
            level="DEBUG"
        )
    
    def start(self) -> None:
        """Start processing incoming samples."""    
        self._is_running = True
    
    def stop(self) -> None:
        """Stop processing and clear buffers."""
        self._is_running = False
    
    def reset(self) -> None:
        """Reset internal state and buffers."""
        self._sample_buffer.clear()
        self._next_window_end_ts = None
        self._is_running = False
    
    def add_sample(self, sample: GazeSample) -> None:
        """
        Add a new gaze sample to the processing buffer.
        
        Args:
            sample: Raw gaze sample from eye tracker.
        """
        self.add_samples([sample])
    
    def add_samples(self, samples: List[GazeSample]) -> None:
        """
        Add raw gaze samples to the processing buffer and emit WindowFeatures
        whenever a complete time window is available.

        AI Generated for POC purposes - may require adjustments.

        Each call may emit zero or more WindowFeatures objects.
        """
        if not self._is_running:
            return

        if not samples:
            return

        # 1. Append incoming samples (assumed time-ordered)
        for sample in samples:
            self._sample_buffer.append(sample)

        # 2. Initialize window end timestamp on first data
        if self._next_window_end_ts is None:
            first_ts = self._sample_buffer[0].timestamp
            self._next_window_end_ts = first_ts + self._config.window_length_seconds

        # 3. Try to produce as many windows as possible
        while True:
            window_end = self._next_window_end_ts
            window_start = window_end - self._config.window_length_seconds

            # Do we have samples covering this window?
            if self._sample_buffer[-1].timestamp < window_end:
                break

            # 4. Extract samples inside window
            window_samples = [
                s for s in self._sample_buffer
                if window_start <= s.timestamp <= window_end
            ]

            # 5. Compute quality metrics
            sample_count = len(window_samples)
            valid_sample_ratio = min(
                sample_count / (self._config.input_sampling_rate_hz * self._config.window_length_seconds), 1.0
            ) # Note : this is calculated before any interpolation or cleaning

            # 6. Compute features (implementation-specific)
            features = self._compute_window_features(window_samples)

            # 7. Emit WindowFeatures
            # TODO: What if valid_sample_ratio < min_valid_sample_ratio?
            window_features = WindowFeatures(
                window_start=window_start,
                window_end=window_end,
                features=features,
                sample_count=sample_count,
                valid_sample_ratio=valid_sample_ratio,
            )

            # 8. Log collected features
            # TODO - figure out what to log here

            # 9. Call output callbacks
            self.call_callbacks(window_features)


            # 10. Advance window
            advance_seconds = self._config.window_length_seconds * (1.0 - self._config.window_overlap_ratio)
            output_frequency_hz = getattr(self._config, "output_frequency_hz", None)
            if output_frequency_hz is not None and output_frequency_hz > 0:
                advance_seconds = 1.0 / output_frequency_hz
            self._next_window_end_ts += advance_seconds

            # 11. Prune old samples (keep only what's needed)
            prune_before_ts = self._next_window_end_ts - self._config.window_length_seconds
            while self._sample_buffer and self._sample_buffer[0].timestamp < prune_before_ts:
                self._sample_buffer.popleft()

        
    def process_raw_data(self, raw_data: RawGazeData) -> List[WindowFeatures]:
        """
        Process a batch of raw gaze data.
        
        Args:
            raw_data: Container with raw gaze samples.
            
        Returns:
            List of computed window features.
        """
        pass  # TODO: Implement batch processing
    
    def get_current_features(self) -> Optional[WindowFeatures]:
        """
        Get the most recently computed window features.
        
        Returns:
            Latest window features or None if not available.
        """
        pass  # TODO: Implement current features getter
    
    def register_output_callback(
        self, callback: Callable[[WindowFeatures], None]
    ) -> None:
        """
        Register a callback to receive output features.
        
        Args:
            callback: Function to call with computed features.
        """
        self._output_callbacks.append(callback)
    
    def unregister_output_callback(
        self, callback: Callable[[WindowFeatures], None]
    ) -> None:
        """
        Unregister a previously registered callback.
        
        Args:
            callback: The callback function to remove.
        """

        if callback in self._output_callbacks:
            self._output_callbacks.remove(callback)
        else:
            self._logger.system(
                "signal_processing_callback_not_found",
                {"callback": str(callback)},
                level="WARNING"
            )

    def call_callbacks(self, features: WindowFeatures) -> None:
        """
        Call all registered output callbacks with the given features.
        
        Args:
            features: The WindowFeatures to pass to callbacks.
        """
        for callback in self._output_callbacks:
            try:
                callback(features)
            except Exception as e:
                self._logger.system(
                    "signal_processing_callback_error",
                    {"error": str(e)},
                    level="ERROR"
                )
    
    # --- Internal methods ---
    
    def _compute_window_features(
        self, samples: List[GazeSample]
    ) -> dict:
        """
        Compute all enabled metrics for a window of samples.
        """

        if not samples:
            return {}

        # 1. Handle missing / invalid samples
        cleaned_samples = self._handle_missing_values(samples)

        features: dict = {}

        # 2. Dispatch enabled metrics # TODO: Optimize later
        for metric in self._config.enabled_metrics:
            try:
                if metric == "pupil_diameter":
                    features.update(
                        self._extract_pupil_metrics(cleaned_samples)
                    )
                elif metric == "data_quality":
                    features.update(
                        self._extract_data_quality_metrics(cleaned_samples)
                    )
                # TODO : Implement other metrics
                # elif metric == "gaze_dispersion":
                #     features.update(
                #         self._extract_gaze_dispersion_metrics(cleaned_samples)
                #     )
                else:
                    self._logger.system(
                        "unknown_metric_requested",
                        {"metric": metric},
                        level="WARNING",
                    )

            except Exception as e:
                self._logger.system(
                    "metric_extraction_failed",
                    {
                        "metric": metric,
                        "error": str(e),
                    },
                    level="WARNING",
                )

        return features
    
    def _handle_missing_values(
        self, samples: List[GazeSample]
    ) -> List[GazeSample]:
        """
        Clean samples by removing invalid entries and optionally interpolating gaps.
        """

        valid_samples = [
            s for s in samples
            if s.left_eye_valid or s.right_eye_valid  # TODO : Define validity criteria
        ]

        if not self._config.interpolate_missing:
            return valid_samples

        # TODO : Implement interpolation logic

        return valid_samples

    
    def _interpolate_gap(
        self, 
        before: GazeSample, 
        after: GazeSample
    ) -> List[GazeSample]:
        """
        Interpolate missing samples between two valid samples.
        
        Args:
            before: Valid sample before the gap.
            after: Valid sample after the gap.
            
        Returns:
            List of interpolated samples.
        """
        pass  # TODO: Implement interpolation

    # -- Metric extraction methods --
    
    def _extract_pupil_metrics(self, samples: List[GazeSample]) -> Dict[str, Any]:
        """
        AI Generated for POC purposes.
        Extract pupil diameter metrics from samples.
        
        """
        def ok(x: Optional[float]) -> bool:
            return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

        total = len(samples)
        if total == 0:
            return {
                "pupil_window_sample_count": 0,
                "pupil_valid_ratio": 0.0,
                "pupil_mean": None,
                "pupil_std": None,
                "pupil_range": None,
                "pupil_slope": None,
                "pupil_mean_abs_vel": None,
            }

        vals: List[float] = []
        times: List[float] = []

        for s in samples:
            l_ok = s.left_eye_valid and ok(s.left_pupil_diameter)
            r_ok = s.right_eye_valid and ok(s.right_pupil_diameter)

            if l_ok and r_ok:
                v = (float(s.left_pupil_diameter) + float(s.right_pupil_diameter)) / 2.0  # type: ignore[arg-type]
            elif l_ok:
                v = float(s.left_pupil_diameter)  # type: ignore[arg-type]
            elif r_ok:
                v = float(s.right_pupil_diameter)  # type: ignore[arg-type]
            else:
                continue

            vals.append(v)
            times.append(float(s.timestamp))

        valid_ratio = len(vals) / total
        if len(vals) == 0:
            return {
                "pupil_window_sample_count": total,
                "pupil_valid_ratio": 0.0,
                "pupil_mean": None,
                "pupil_std": None,
                "pupil_range": None,
                "pupil_slope": None,
                "pupil_mean_abs_vel": None,
            }

        # mean / std / range
        n = len(vals)
        s = sum(vals)
        mean = s / n
        var = sum((v - mean) ** 2 for v in vals) / n  # population variance
        std = math.sqrt(var)
        vmin = min(vals)
        vmax = max(vals)
        vrange = vmax - vmin

        # slope (least squares) diameter per second
        slope = None
        if n >= 2:
            mt = sum(times) / n
            mv = mean
            num = sum((t - mt) * (v - mv) for t, v in zip(times, vals))
            den = sum((t - mt) ** 2 for t in times)
            slope = (num / den) if den > 0 else None

        # mean abs velocity
        mean_abs_vel = None
        if n >= 2:
            acc = 0.0
            k = 0
            for i in range(1, n):
                dt = times[i] - times[i - 1]
                if dt <= 0:
                    continue
                acc += abs((vals[i] - vals[i - 1]) / dt)
                k += 1
            mean_abs_vel = (acc / k) if k > 0 else None

        return {
            "pupil_window_sample_count": total,
            "pupil_valid_ratio": valid_ratio,
            "pupil_mean": mean,
            "pupil_std": std,
            "pupil_range": vrange,
            "pupil_slope": slope,
            "pupil_mean_abs_vel": mean_abs_vel,
        }


    def _extract_data_quality_metrics(self, samples: List[GazeSample]) -> Dict[str, Any]:
        """
        AI Generated for POC purposes.
        Extract data quality metrics from samples.

        """
        def finite(x: Optional[float]) -> bool:
            return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

        n = len(samples)
        if n == 0:
            return {
                "dq_sample_count": 0,
                "dq_valid_ratio_left": 0.0,
                "dq_valid_ratio_right": 0.0,
                "dq_valid_ratio_any": 0.0,
                "dq_valid_ratio_both": 0.0,
                "dq_longest_invalid_run_ms": 0.0,
                "dq_gap_count_over_100ms": 0,
                "dq_mean_dt_ms": None,
                "dq_dt_jitter_ms": None,
            }

        left_ok = 0
        right_ok = 0
        any_ok = 0
        both_ok = 0

        # dt metrics
        dts = []
        for i in range(1, n):
            dt = (samples[i].timestamp - samples[i-1].timestamp)
            if dt > 0:
                dts.append(dt)

        mean_dt = (sum(dts) / len(dts)) if dts else None
        dt_jitter = None
        if mean_dt is not None and dts:
            var = sum((dt - mean_dt) ** 2 for dt in dts) / len(dts)
            dt_jitter = math.sqrt(var)

        # invalid-run / gaps (based on "any eye valid")
        longest_invalid_run = 0.0
        current_invalid_run = 0.0
        gap_count_over_100ms = 0

        for i, s in enumerate(samples):
            l = bool(s.left_eye_valid) and finite(s.left_eye_x) and finite(s.left_eye_y)
            r = bool(s.right_eye_valid) and finite(s.right_eye_x) and finite(s.right_eye_y)

            if l:
                left_ok += 1
            if r:
                right_ok += 1
            if l or r:
                any_ok += 1
            if l and r:
                both_ok += 1

            # invalid runs in time
            if i > 0:
                dt = samples[i].timestamp - samples[i-1].timestamp
                if dt > 0.100:  # 100 ms
                    gap_count_over_100ms += 1

            if not (l or r):
                # accumulate time in invalid streak (approx by dt to next sample)
                if i > 0:
                    dt = samples[i].timestamp - samples[i-1].timestamp
                    if dt > 0:
                        current_invalid_run += dt
                longest_invalid_run = max(longest_invalid_run, current_invalid_run)
            else:
                current_invalid_run = 0.0

        return {
            "dq_sample_count": n,
            "dq_valid_ratio_left": left_ok / n,
            "dq_valid_ratio_right": right_ok / n,
            "dq_valid_ratio_any": any_ok / n,
            "dq_valid_ratio_both": both_ok / n,
            "dq_longest_invalid_run_ms": longest_invalid_run * 1000.0,
            "dq_gap_count_over_100ms": gap_count_over_100ms,
            "dq_mean_dt_ms": (mean_dt * 1000.0) if mean_dt is not None else None,
            "dq_dt_jitter_ms": (dt_jitter * 1000.0) if dt_jitter is not None else None,
        }
