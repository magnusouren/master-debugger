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
from typing import Any, Dict, Optional, List, Callable, Tuple
from collections import deque
from backend.services.logger_service import LoggerService


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
    
    def __init__(
        self,
        config: Optional[SignalProcessingConfig] = None,
        logger: Optional[LoggerService] = None,
    ):
        """
        Initialize the Signal Processing layer.
        
        Args:
            config: Configuration for signal processing parameters.
            logger: Logger instance for recording events.
        """
        self._config = config or SignalProcessingConfig()
        self._sample_buffer: deque[GazeSample] = deque()
        self._output_callbacks: List[Callable[[WindowFeatures], None]] = []
        self._last_output_time: float = 0.0
        self._is_running: bool = False
        
        # Use provided logger or create fallback
        if logger is None:
            from backend.services.logger_service import get_logger
            logger = get_logger()
        self._logger = logger
        
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
                "min_pupil_diameter_mm": config.min_pupil_diameter_mm,
                "max_pupil_diameter_mm": config.max_pupil_diameter_mm,
                "min_gaze_coordinate": config.min_gaze_coordinate,
                "max_gaze_coordinate": config.max_gaze_coordinate,
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

            # 7. Check data quality and emit WindowFeatures
            is_low_quality = valid_sample_ratio < self._config.min_valid_sample_ratio

            if is_low_quality:
                self._logger.system(
                    "low_quality_window",
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "valid_sample_ratio": valid_sample_ratio,
                        "min_required": self._config.min_valid_sample_ratio,
                        "sample_count": sample_count,
                    },
                    level="WARNING"
                )

            window_features = WindowFeatures(
                window_start=window_start,
                window_end=window_end,
                features=features,
                sample_count=sample_count,
                valid_sample_ratio=valid_sample_ratio,
                enabled_metrics=list(self._config.enabled_metrics)
            )

            # 8. Call output callbacks (emit even if low quality - let downstream decide)
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
                elif metric == "gaze_dispersion":
                    features.update(
                        self._extract_gaze_dispersion_metrics(cleaned_samples)
                    )
                elif metric == "fixation_duration":
                    features.update(
                        self._extract_fixation_metrics(cleaned_samples)
                    )
                elif metric == "saccade_amplitude":
                    features.update(
                        self._extract_saccade_metrics(cleaned_samples)
                    )
                elif metric == "blink_rate":
                    features.update(
                        self._extract_blink_rate(samples)
                    )
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
    
    def _is_sample_valid(self, sample: GazeSample) -> bool:
        """
        Determine if a gaze sample meets validity criteria.

        A sample is considered valid if at least one eye has:
        - Valid flag set
        - Finite gaze coordinates within configured bounds
        - Finite pupil diameter within physiological range (if available)

        Args:
            sample: The gaze sample to validate.

        Returns:
            True if the sample is valid, False otherwise.
        """
        def is_finite(x: Optional[float]) -> bool:
            """Check if value is not None and is a finite number."""
            return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

        def is_in_bounds(x: Optional[float], min_val: float, max_val: float) -> bool:
            """Check if value is finite and within bounds."""
            return is_finite(x) and min_val <= x <= max_val  # type: ignore[arg-type, operator]

        # Check left eye validity
        left_valid = False
        if sample.left_eye_valid:
            # Check gaze coordinates
            gaze_valid = (
                is_in_bounds(
                    sample.left_eye_x,
                    self._config.min_gaze_coordinate,
                    self._config.max_gaze_coordinate
                ) and
                is_in_bounds(
                    sample.left_eye_y,
                    self._config.min_gaze_coordinate,
                    self._config.max_gaze_coordinate
                )
            )

            # Check pupil diameter if available
            pupil_valid = True
            if sample.left_pupil_diameter is not None:
                pupil_valid = is_in_bounds(
                    sample.left_pupil_diameter,
                    self._config.min_pupil_diameter_mm,
                    self._config.max_pupil_diameter_mm
                )

            left_valid = gaze_valid and pupil_valid

        # Check right eye validity
        right_valid = False
        if sample.right_eye_valid:
            # Check gaze coordinates
            gaze_valid = (
                is_in_bounds(
                    sample.right_eye_x,
                    self._config.min_gaze_coordinate,
                    self._config.max_gaze_coordinate
                ) and
                is_in_bounds(
                    sample.right_eye_y,
                    self._config.min_gaze_coordinate,
                    self._config.max_gaze_coordinate
                )
            )

            # Check pupil diameter if available
            pupil_valid = True
            if sample.right_pupil_diameter is not None:
                pupil_valid = is_in_bounds(
                    sample.right_pupil_diameter,
                    self._config.min_pupil_diameter_mm,
                    self._config.max_pupil_diameter_mm
                )

            right_valid = gaze_valid and pupil_valid

        return left_valid or right_valid

    def _handle_missing_values(
        self, samples: List[GazeSample]
    ) -> List[GazeSample]:
        """
        Clean samples by removing invalid entries and optionally interpolating gaps.
        """
        if not samples:
            return []

        # Filter samples using robust validity checking
        initial_count = len(samples)
        valid_samples = [s for s in samples if self._is_sample_valid(s)]
        filtered_count = len(valid_samples)

        # Log if significant data was filtered
        if initial_count > 0:
            rejection_ratio = (initial_count - filtered_count) / initial_count
            if rejection_ratio > 0.2:  # More than 20% rejected
                self._logger.system(
                    "high_sample_rejection_rate",
                    {
                        "initial_count": initial_count,
                        "valid_count": filtered_count,
                        "rejection_ratio": rejection_ratio,
                    },
                    level="WARNING"
                )

        if not self._config.interpolate_missing:
            return valid_samples

        # TODO: Implement interpolation logic

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

    # -- Gaze position / I-VT helpers --

    def _get_gaze_positions(self, samples: List[GazeSample]) -> List[Tuple[float, float, float]]:
        """Extract binocular gaze positions as (timestamp, x, y) from valid samples."""
        positions: List[Tuple[float, float, float]] = []
        for s in samples:
            l_ok = s.left_eye_valid and s.left_eye_x is not None and s.left_eye_y is not None
            r_ok = s.right_eye_valid and s.right_eye_x is not None and s.right_eye_y is not None

            if l_ok and r_ok:
                x = (float(s.left_eye_x) + float(s.right_eye_x)) / 2.0  # type: ignore[arg-type]
                y = (float(s.left_eye_y) + float(s.right_eye_y)) / 2.0  # type: ignore[arg-type]
            elif l_ok:
                x = float(s.left_eye_x)  # type: ignore[arg-type]
                y = float(s.left_eye_y)  # type: ignore[arg-type]
            elif r_ok:
                x = float(s.right_eye_x)  # type: ignore[arg-type]
                y = float(s.right_eye_y)  # type: ignore[arg-type]
            else:
                continue

            positions.append((s.timestamp, x, y))
        return positions

    def _segment_by_ivt(
        self,
        positions: List[Tuple[float, float, float]],
    ) -> List[Tuple[bool, List[Tuple[float, float, float]]]]:
        """
        Segment gaze positions into fixations and saccades using I-VT
        (velocity-threshold classification).

        Returns a list of (is_saccade, positions_in_segment) tuples in
        chronological order.
        """
        threshold = self._config.saccade_velocity_threshold

        if len(positions) < 2:
            return [(False, positions)] if positions else []

        # Label each sample based on the velocity from the *previous* sample.
        # labels[i] for i >= 1 corresponds to the velocity between positions[i-1]
        # and positions[i].  The first position inherits the label of the first
        # velocity measurement.
        labels: List[bool] = []
        for i in range(1, len(positions)):
            dt = positions[i][0] - positions[i - 1][0]
            if dt <= 0:
                labels.append(False)
                continue
            dx = positions[i][1] - positions[i - 1][1]
            dy = positions[i][2] - positions[i - 1][2]
            vel = math.sqrt(dx * dx + dy * dy) / dt
            labels.append(vel >= threshold)

        labels.insert(0, labels[0])  # first position inherits first label

        # Extract contiguous segments
        segments: List[Tuple[bool, List[Tuple[float, float, float]]]] = []
        seg_start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[seg_start]:
                segments.append((labels[seg_start], positions[seg_start:i]))
                seg_start = i
        segments.append((labels[seg_start], positions[seg_start:]))

        return segments

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

    def _extract_gaze_dispersion_metrics(self, samples: List[GazeSample]) -> Dict[str, Any]:
        """
        Extract gaze dispersion metrics (spread/variability of gaze points).

        Higher dispersion indicates less stable gaze, potentially more confusion or searching.
        """
        positions = self._get_gaze_positions(samples)
        n = len(positions)

        if n < 2:
            return {
                "gaze_disp_sample_count": n,
                "gaze_disp_x_std": None,
                "gaze_disp_y_std": None,
                "gaze_disp_total": None,
                "gaze_disp_max_dist": None,
            }

        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        var_x = sum((x - mean_x) ** 2 for x in xs) / n
        var_y = sum((y - mean_y) ** 2 for y in ys) / n

        return {
            "gaze_disp_sample_count": n,
            "gaze_disp_x_std": math.sqrt(var_x),
            "gaze_disp_y_std": math.sqrt(var_y),
            "gaze_disp_total": math.sqrt(var_x + var_y),
            "gaze_disp_max_dist": max(
                math.sqrt((x - mean_x) ** 2 + (y - mean_y) ** 2)
                for x, y in zip(xs, ys)
            ),
        }

    def _extract_fixation_metrics(self, samples: List[GazeSample]) -> Dict[str, Any]:
        """
        Extract fixation metrics using I-VT (velocity threshold) algorithm.

        Fixations are periods of relatively stationary gaze.  Longer fixations
        may indicate processing difficulty.
        """
        positions = self._get_gaze_positions(samples)

        if len(positions) < 2:
            return {
                "fixation_count": 0,
                "fixation_mean_duration_ms": None,
                "fixation_max_duration_ms": None,
                "fixation_mean_dispersion": None,
            }

        segments = self._segment_by_ivt(positions)
        min_dur_ms = self._config.min_fixation_duration_ms

        durations: List[float] = []
        dispersions: List[float] = []

        for is_saccade, seg_positions in segments:
            if is_saccade or len(seg_positions) < 2:
                continue

            dur_ms = (seg_positions[-1][0] - seg_positions[0][0]) * 1000.0
            if dur_ms < min_dur_ms:
                continue

            durations.append(dur_ms)

            # Spatial dispersion within this fixation
            n = len(seg_positions)
            mx = sum(p[1] for p in seg_positions) / n
            my = sum(p[2] for p in seg_positions) / n
            disp = math.sqrt(
                sum((p[1] - mx) ** 2 for p in seg_positions) / n +
                sum((p[2] - my) ** 2 for p in seg_positions) / n
            )
            dispersions.append(disp)

        count = len(durations)
        return {
            "fixation_count": count,
            "fixation_mean_duration_ms": sum(durations) / count if count > 0 else None,
            "fixation_max_duration_ms": max(durations) if count > 0 else None,
            "fixation_mean_dispersion": sum(dispersions) / len(dispersions) if dispersions else None,
        }

    def _extract_saccade_metrics(self, samples: List[GazeSample]) -> Dict[str, Any]:
        """
        Extract saccade metrics using I-VT (velocity threshold) algorithm.

        Saccades are rapid eye movements between fixations.  Amplitude is the
        Euclidean distance (normalized screen units) from the end of the
        preceding fixation to the start of the following fixation.  Only
        complete saccades (with a fixation on both sides) are counted.
        """
        positions = self._get_gaze_positions(samples)

        if len(positions) < 2:
            return {
                "saccade_count": 0,
                "saccade_mean_amplitude": None,
                "saccade_max_amplitude": None,
                "saccade_mean_duration_ms": None,
            }

        segments = self._segment_by_ivt(positions)

        amplitudes: List[float] = []
        durations: List[float] = []

        # Walk segments tracking the cumulative position index so we can
        # reference the boundary samples of neighbouring fixations.
        pos_idx = 0
        prev_fixation_end_idx: Optional[int] = None

        for is_saccade, seg_positions in segments:
            seg_len = len(seg_positions)

            if not is_saccade:
                prev_fixation_end_idx = pos_idx + seg_len - 1
            else:
                next_fixation_start_idx = pos_idx + seg_len  # first sample after saccade

                if prev_fixation_end_idx is not None and next_fixation_start_idx < len(positions):
                    start = positions[prev_fixation_end_idx]
                    end = positions[next_fixation_start_idx]

                    dx = end[1] - start[1]
                    dy = end[2] - start[2]
                    amplitudes.append(math.sqrt(dx * dx + dy * dy))

                    # Duration: from saccade onset to next fixation start
                    durations.append(
                        (positions[next_fixation_start_idx][0] - positions[pos_idx][0]) * 1000.0
                    )

            pos_idx += seg_len

        count = len(amplitudes)
        return {
            "saccade_count": count,
            "saccade_mean_amplitude": sum(amplitudes) / count if count > 0 else None,
            "saccade_max_amplitude": max(amplitudes) if count > 0 else None,
            "saccade_mean_duration_ms": sum(durations) / count if count > 0 else None,
        }
    
    # Made the function, but we are not using it for calculating in reactive tool currently.
    def _extract_blink_rate(self, samples: List[GazeSample]) -> Dict[str, Any]:
        """
        Extract blink metrics by detecting validity gaps in both eyes.

        A blink is a period where both eyes report invalid, lasting between
        100–400 ms (physiological blink range).  This method must receive the
        raw (uncleaned) samples so that validity transitions are visible.
        """
        if not samples:
            return {
                "blink_count": 0,
                "blink_rate_per_min": 0.0,
                "blink_mean_duration_ms": None,
            }

        blink_durations: List[float] = []
        in_blink = False
        blink_start: float = 0.0

        for s in samples:
            both_invalid = not (s.left_eye_valid or s.right_eye_valid)

            if both_invalid and not in_blink:
                in_blink = True
                blink_start = s.timestamp
            elif not both_invalid and in_blink:
                dur_ms = (s.timestamp - blink_start) * 1000.0
                if 100.0 <= dur_ms <= 400.0:
                    blink_durations.append(dur_ms)
                in_blink = False

        # Extrapolate rate to per-minute based on observed window duration
        window_duration_s = samples[-1].timestamp - samples[0].timestamp
        window_duration_min = window_duration_s / 60.0 if window_duration_s > 0 else 1.0

        count = len(blink_durations)
        return {
            "blink_count": count,
            "blink_rate_per_min": count / window_duration_min if window_duration_min > 0 else 0.0,
            "blink_mean_duration_ms": sum(blink_durations) / count if count > 0 else None,
        }
