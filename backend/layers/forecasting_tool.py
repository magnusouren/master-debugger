"""
Forecasting Tool (Proactive mode only)

Position: Between Signal Processing and Reactive Tool
Input: Window-based features (e.g., 2–10 Hz)
Output: Predicted window-based features (same format)
Configuration: Prediction horizon (seconds into the future), Update rate
"""
from __future__ import annotations

from typing import Optional, List, Callable, Any, Tuple
from collections import deque
from pathlib import Path
import time

import numpy as np

from backend.services.logger_service import LoggerService
from backend.types import (
    WindowFeatures,
    PredictedFeatures,
    ForecastingConfig,
)
from backend.models.xgboost_forecaster import XGBoostForecaster
from backend.models.forecast_feature_schema import TARGET_COLUMNS


DEFAULT_INPUT_COLUMNS = [
    "pupil_ipa",
    "fixation_mean_duration_ms",
]


class ForecastingTool:
    """
    Predicts future eye-tracking features for proactive intervention.
    """

    def __init__(
        self,
        config: Optional[ForecastingConfig] = None,
        logger: Optional[LoggerService] = None,
    ):
        self._config = config or ForecastingConfig()
        self._feature_history: deque[WindowFeatures] = deque()
        self._forecaster: Optional[XGBoostForecaster] = None
        self._output_callbacks: List[Callable[[PredictedFeatures], None]] = []

        self._is_enabled: bool = False
        self._last_prediction_time: float = 0.0
        self._forecast_counter: int = 0
        self._pred_window_counter: int = 0
        self._last_warmup_status: Optional[str] = None
        self._id_prefix: str = f"fc{int(time.time())}"

        if logger is None:
            from backend.services.logger_service import get_logger
            logger = get_logger()
        self._logger = logger

        if self._config.model_path:
            self.load_model(self._config.model_path)
        else:
            self._logger.system(
                "forecasting_tool_no_model_configured",
                {},
                level="WARNING",
            )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def configure(self, config: ForecastingConfig) -> None:
        self._config = config

    def enable(self) -> None:
        if not self._has_loaded_model():
            self.load_model(self._config.model_path)
        self._is_enabled = True

    def disable(self) -> None:
        self._is_enabled = False

    def is_enabled(self) -> bool:
        return self._is_enabled

    def load_model(self, model_path: Optional[str] = None) -> bool:
        try:
            self._forecaster = XGBoostForecaster()
            path = Path(model_path) if model_path else None
            success = self._forecaster.load_model(path)

            if success:
                self._logger.system(
                    "forecasting_tool_model_loaded",
                    {
                        "model_path": str(model_path),
                        "info": self._forecaster.get_model_info(),
                    },
                )
            else:
                self._logger.system(
                    "forecasting_tool_model_load_failed",
                    {"model_path": str(model_path)},
                    level="WARNING",
                )

            return success
        except Exception as e:
            self._logger.system(
                "forecasting_tool_model_load_error",
                {
                    "model_path": str(model_path),
                    "error": str(e),
                },
                level="ERROR",
            )
            return False

    def unload_model(self) -> None:
        self._forecaster = None
        self._logger.system("forecasting_tool_model_unloaded", {})

    def add_features(self, features: WindowFeatures) -> None:
        self._append_to_history(features)
        self._trim_history(features.window_end)

        if not self._is_enabled:
            return

        if not self._should_update_prediction():
            return

        prediction = self.predict()
        if prediction is None:
            return

        if prediction.confidence < self._config.min_confidence_threshold:
            return

        for callback in list(self._output_callbacks):
            try:
                callback(prediction)
            except Exception as e:
                self._logger.system(
                    "forecasting_tool_callback_error",
                    {"error": str(e)},
                    level="ERROR",
                )

    def predict(self) -> Optional[PredictedFeatures]:
        if not self._config.prediction_horizon_seconds:
            self._logger.system(
                "forecasting_tool_no_horizon_configured",
                {},
                level="WARNING",
            )
            return None

        return self.predict_at_horizon(self._config.prediction_horizon_seconds)

    def predict_at_horizon(self, horizon_seconds: float) -> Optional[PredictedFeatures]:
        started_at = time.perf_counter()
        forecast_id = self._next_forecast_id()

        input_sequence = self._prepare_input_sequence()
        if not input_sequence:
            return None

        prediction = self._run_model_inference(
            input_sequence=input_sequence,
            forecast_id=forecast_id,
            horizon_seconds=horizon_seconds,
        )

        if prediction is not None:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self._logger.system(
                "forecasting_tool_prediction_latency",
                {
                    "latency_ms": round(elapsed_ms, 3),
                    "history_windows": len(input_sequence),
                    "horizon_seconds": horizon_seconds,
                },
                level="DEBUG",
            )

        return prediction

    def register_output_callback(
        self,
        callback: Callable[[PredictedFeatures], None],
    ) -> None:
        self._output_callbacks.append(callback)

    def unregister_output_callback(
        self,
        callback: Callable[[PredictedFeatures], None],
    ) -> None:
        if callback in self._output_callbacks:
            self._output_callbacks.remove(callback)

    def reset(self) -> None:
        self._feature_history.clear()
        self._last_prediction_time = 0.0
        self._last_warmup_status = None
        self._forecast_counter = 0
        self._pred_window_counter = 0

    # -------------------------------------------------------------------------
    # Internal: history / warmup
    # -------------------------------------------------------------------------

    def _append_to_history(self, features: WindowFeatures) -> None:
        self._feature_history.append(features)

    def _trim_history(self, latest_window_end: float) -> None:
        # Keep time-based retention, but never below model-required history windows.
        min_seconds = float(self._config.history_window_seconds)
        required_windows = self._compute_required_history()

        if required_windows > 1 and len(self._feature_history) >= 2:
            step_seconds = self._estimate_window_step_seconds()
            if step_seconds > 0.0:
                required_seconds = (required_windows - 1) * step_seconds
                min_seconds = max(min_seconds, required_seconds)

        cutoff_time = latest_window_end - min_seconds
        while self._feature_history and self._feature_history[0].window_end < cutoff_time:
            self._feature_history.popleft()

    def _estimate_window_step_seconds(self) -> float:
        if len(self._feature_history) < 2:
            return 0.0

        latest = self._feature_history[-1]
        prev = self._feature_history[-2]

        step = float(latest.window_end - prev.window_end)
        if step > 0.0:
            return step

        duration = float(latest.window_end - latest.window_start)
        return duration if duration > 0.0 else 0.0

    def _prepare_input_sequence(self) -> Optional[List[WindowFeatures]]:
        required_history = self._compute_required_history()
        can_predict = len(self._feature_history) >= required_history

        self._log_warmup_status(
            can_predict=can_predict,
            required_windows=required_history,
        )

        if not can_predict:
            return None

        return list(self._feature_history)

    def _compute_required_history(self) -> int:
        if self._has_loaded_model():
            return max(1, int(getattr(self._forecaster, "history_size", 5)))
        return 5

    def _current_window_duration(self) -> float:
        if not self._feature_history:
            return 0.0
        latest = self._feature_history[-1]
        return max(0.0, latest.window_end - latest.window_start)

    def _log_warmup_status(self, can_predict: bool, required_windows: int) -> None:
        if not self._is_enabled:
            return

        status = "ready" if can_predict else "warming_up"
        buffer_windows = len(self._feature_history)
        buffer_fill_ratio = (
            0.0 if required_windows <= 0 else min(1.0, buffer_windows / required_windows)
        )

        available_history_seconds = 0.0
        if self._feature_history:
            available_history_seconds = max(
                0.0,
                self._feature_history[-1].window_end - self._feature_history[0].window_start,
            )

        if status == self._last_warmup_status and status == "ready":
            return

        self._logger.experiment(
            "forecast_warmup_status_logged",
            {
                "status": status,
                "buffer_fill_ratio": round(buffer_fill_ratio, 3),
                "buffer_windows": buffer_windows,
                "required_windows": required_windows,
                "available_history_seconds": round(available_history_seconds, 3),
                "can_predict": can_predict,
            },
            level="INFO",
        )
        self._last_warmup_status = status

    # -------------------------------------------------------------------------
    # Internal: ids / helpers
    # -------------------------------------------------------------------------

    def _next_forecast_id(self) -> str:
        self._forecast_counter += 1
        return f"{self._id_prefix}-f{self._forecast_counter:06d}"

    def _next_pred_window_id(self, forecast_id: str) -> str:
        self._pred_window_counter += 1
        return f"{forecast_id}-w{self._pred_window_counter:04d}"

    def _has_loaded_model(self) -> bool:
        return self._forecaster is not None and self._forecaster.is_loaded()

    # -------------------------------------------------------------------------
    # Internal: schema resolution
    # -------------------------------------------------------------------------

    def _resolve_model_schema(
        self,
        input_sequence: List[WindowFeatures],
    ) -> Tuple[int, List[str], List[str]]:
        metadata = getattr(self._forecaster, "_metadata", {}) or {}

        history_size = int(
            metadata.get(
                "history_window_size",
                getattr(self._forecaster, "history_size", 0) or len(input_sequence),
            )
        )

        input_columns = list(
            metadata.get(
                "input_columns",
                getattr(self._forecaster, "feature_names", []),
            )
        )
        if not input_columns:
            input_columns = list(DEFAULT_INPUT_COLUMNS)

        target_columns = list(metadata.get("target_columns", TARGET_COLUMNS))
        return history_size, input_columns, target_columns

    def _select_recent_windows(
        self,
        input_sequence: List[WindowFeatures],
        required_history_size: int,
    ) -> Optional[List[WindowFeatures]]:
        if len(input_sequence) < required_history_size:
            self._logger.system(
                "forecasting_tool_not_enough_history",
                {
                    "have_windows": len(input_sequence),
                    "need_windows": required_history_size,
                },
                level="WARNING",
            )

        return list(input_sequence)[-required_history_size:]

    # -------------------------------------------------------------------------
    # Internal: feature flattening / model output parsing
    # -------------------------------------------------------------------------

    @staticmethod
    def _safe_window_stat(values: np.ndarray, fn: Callable[[np.ndarray], float]) -> float:
        if values.size == 0:
            return 0.0
        return float(fn(values))

    @staticmethod
    def _safe_window_slope(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0

        x = np.arange(values.size, dtype=np.float32)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(values))
        denom = float(np.sum((x - x_mean) ** 2))
        if denom == 0.0:
            return 0.0

        numer = float(np.sum((x - x_mean) * (values - y_mean)))
        return numer / denom

    def _build_summary_features(
        self,
        history_matrix: np.ndarray,
        input_columns: List[str],
    ) -> List[float]:
        summary_values: List[float] = []

        for col_idx, _col_name in enumerate(input_columns):
            series = history_matrix[:, col_idx]

            last_3 = series[-3:]
            last_5 = series[-5:]
            last_10 = series[-10:]
            prev_5 = series[-10:-5] if series.shape[0] >= 10 else np.array([], dtype=series.dtype)

            feature_values = [
                float(series[-1]),
                self._safe_window_stat(last_3, np.mean),
                self._safe_window_stat(last_5, np.mean),
                self._safe_window_stat(last_10, np.mean),
                self._safe_window_stat(last_5, np.std),
                self._safe_window_stat(last_10, np.std),
                self._safe_window_stat(last_10, np.min),
                self._safe_window_stat(last_10, np.max),
                (
                    self._safe_window_stat(last_10, np.max)
                    - self._safe_window_stat(last_10, np.min)
                ) if last_10.size > 0 else 0.0,
                float(series[-1] - series[-2]) if series.shape[0] >= 2 else 0.0,
                (
                    self._safe_window_stat(last_5, np.mean)
                    - self._safe_window_stat(prev_5, np.mean)
                ) if prev_5.size > 0 else 0.0,
                self._safe_window_slope(last_5),
                self._safe_window_slope(last_10),
            ]

            summary_values.extend(float(v) for v in feature_values)

        return summary_values

    def _flatten_windows(
        self,
        windows: List[WindowFeatures],
        input_columns: List[str],
    ) -> Tuple[List[float], List[str]]:
        history_rows: List[List[float]] = []
        missing_inputs: List[str] = []

        for window in windows:
            feature_dict = window.features or {}
            row_values: List[float] = []
            for col in input_columns:
                val = feature_dict.get(col)
                if val is None:
                    missing_inputs.append(col)
                row_values.append(self._forecaster._safe_float(val))
            history_rows.append(row_values)

        history_matrix = np.asarray(history_rows, dtype=np.float32)
        flat_features = history_matrix.flatten().tolist()

        metadata = getattr(self._forecaster, "_metadata", {}) or {}
        expected_feature_count = int(
            metadata.get("n_features", len(metadata.get("feature_names", [])) or len(flat_features))
        )

        if expected_feature_count > len(flat_features):
            flat_features.extend(self._build_summary_features(history_matrix, input_columns))

        return flat_features, missing_inputs

    def _predict_raw_array(
        self,
        flat_features: List[float],
        recent_windows: List[WindowFeatures],
        input_columns: List[str],
        target_columns: List[str],
        model_history_size: int,
        full_history_size: int,
    ) -> Optional[np.ndarray]:
        try:
            return np.asarray(self._forecaster._model.predict(np.array([flat_features])))
        except Exception as e:
            last_window = recent_windows[-1]
            self._logger.system(
                "forecasting_tool_inference_error",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "history_windows": full_history_size,
                    "model_history_size": model_history_size,
                    "input_columns": input_columns,
                    "target_columns": target_columns,
                    "flat_feature_count": len(flat_features),
                    "last_window_keys": sorted(list((last_window.features or {}).keys())),
                },
                level="ERROR",
            )
            return None

    # -------------------------------------------------------------------------
    # Internal: feature flattening / model output parsing
    # -------------------------------------------------------------------------

    def _flatten_windows(
        self,
        windows: List[WindowFeatures],
        input_columns: List[str],
    ) -> Tuple[List[float], List[str]]:
        history_rows: List[List[float]] = []
        missing_inputs: List[str] = []

        for window in windows:
            feature_dict = window.features or {}
            row_values: List[float] = []
            for col in input_columns:
                val = feature_dict.get(col)
                if val is None:
                    missing_inputs.append(col)
                row_values.append(self._forecaster._safe_float(val))
            history_rows.append(row_values)

        history_matrix = np.asarray(history_rows, dtype=np.float32)
        flat_features = history_matrix.flatten().tolist()

        metadata = getattr(self._forecaster, "_metadata", {}) or {}
        expected_feature_count = int(
            metadata.get("n_features", len(metadata.get("feature_names", [])) or len(flat_features))
        )

        if expected_feature_count > len(flat_features):
            flat_features.extend(self._build_summary_features(history_matrix, input_columns))

        return flat_features, missing_inputs

    def _predict_raw_array(
        self,
        flat_features: List[float],
        recent_windows: List[WindowFeatures],
        input_columns: List[str],
        target_columns: List[str],
        model_history_size: int,
        full_history_size: int,
    ) -> Optional[np.ndarray]:
        try:
            return np.asarray(self._forecaster._model.predict(np.array([flat_features])))
        except Exception as e:
            last_window = recent_windows[-1]
            self._logger.system(
                "forecasting_tool_inference_error",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "history_windows": full_history_size,
                    "model_history_size": model_history_size,
                    "input_columns": input_columns,
                    "target_columns": target_columns,
                    "flat_feature_count": len(flat_features),
                    "last_window_keys": sorted(list((last_window.features or {}).keys())),
                },
                level="ERROR",
            )
            return None

    def _extract_prediction_row(
        self,
        pred_arr: np.ndarray,
        target_columns: List[str],
        missing_inputs: List[str],
    ) -> Optional[np.ndarray]:
        self._logger.system(
            "forecasting_tool_prediction_shape",
            {
                "shape": list(pred_arr.shape),
                "target_columns": target_columns,
                "missing_inputs": sorted(set(missing_inputs)),
            },
            level="DEBUG",
        )

        if pred_arr.ndim == 2 and pred_arr.shape[0] >= 1:
            pred_row = pred_arr[0]
        elif pred_arr.ndim == 1:
            pred_row = pred_arr
        else:
            self._logger.system(
                "forecasting_tool_unexpected_prediction_shape",
                {"shape": list(pred_arr.shape)},
                level="ERROR",
            )
            return None

        if len(pred_row) < len(target_columns):
            self._logger.system(
                "forecasting_tool_prediction_missing_targets",
                {
                    "expected": target_columns,
                    "got_len": len(pred_row),
                },
                level="ERROR",
            )
            return None

        return pred_row

    def _build_predicted_components(
        self,
        pred_row: np.ndarray,
        target_columns: List[str],
        latest_window: WindowFeatures,
    ) -> dict:
        metadata = getattr(self._forecaster, "_metadata", {}) or {}
        target_type = str(metadata.get("target_type", "absolute")).lower()

        latest_features = latest_window.features or {}
        predicted: dict = {}

        for idx, col in enumerate(target_columns):
            raw_pred = float(pred_row[idx])

            if target_type == "delta":
                current_value = self._forecaster._safe_float(latest_features.get(col))
                predicted_value = current_value + raw_pred
            else:
                predicted_value = raw_pred

            predicted[col] = float(predicted_value)

        return predicted
    # -------------------------------------------------------------------------
    # Internal: prediction object construction
    # -------------------------------------------------------------------------

    def _estimate_prediction_confidence(
        self,
        missing_inputs: List[str],
        expected_input_count: int,
    ) -> float:
        if expected_input_count <= 0:
            return 0.0

        missing_ratio = len(missing_inputs) / expected_input_count
        confidence = 1.0 - missing_ratio
        return float(max(0.0, min(1.0, confidence)))

    def _build_predicted_features(
        self,
        latest_window: WindowFeatures,
        predicted_components: dict,
        forecast_id: str,
        horizon_seconds: float,
        confidence: float,
    ) -> PredictedFeatures:
        window_duration = latest_window.window_end - latest_window.window_start

        predicted = PredictedFeatures(
            prediction_timestamp=latest_window.window_end,
            target_window_start=latest_window.window_end + horizon_seconds,
            target_window_end=latest_window.window_end + horizon_seconds + window_duration,
            horizon_seconds=horizon_seconds,
            forecast_id=forecast_id,
            window_id=self._next_pred_window_id(forecast_id),
            features=predicted_components,
            enabled_metrics=(latest_window.enabled_metrics or []).copy(),
            confidence=confidence,
            uncertainty={},
        )

        setattr(predicted, "sample_count", latest_window.sample_count)
        setattr(predicted, "valid_sample_ratio", latest_window.valid_sample_ratio)
        return predicted

    # -------------------------------------------------------------------------
    # Internal: core inference
    # -------------------------------------------------------------------------

    def _run_model_inference(
        self,
        input_sequence: List[WindowFeatures],
        forecast_id: str,
        horizon_seconds: float,
    ) -> Optional[PredictedFeatures]:
        if not self._has_loaded_model():
            self._logger.system(
                "forecasting_tool_no_model",
                {"message": "No model loaded for inference"},
                level="WARNING",
            )
            return None

        model_history_size, input_columns, target_columns = self._resolve_model_schema(
            input_sequence
        )

        recent_windows = self._select_recent_windows(
            input_sequence=input_sequence,
            required_history_size=model_history_size,
        )
        if recent_windows is None:
            return None

        flat_features, missing_inputs = self._flatten_windows(
            windows=recent_windows,
            input_columns=input_columns,
        )

        infer_started_at = time.perf_counter()
        pred_arr = self._predict_raw_array(
            flat_features=flat_features,
            recent_windows=recent_windows,
            input_columns=input_columns,
            target_columns=target_columns,
            model_history_size=model_history_size,
            full_history_size=len(input_sequence),
        )
        if pred_arr is None:
            return None

        pred_row = self._extract_prediction_row(
            pred_arr=pred_arr,
            target_columns=target_columns,
            missing_inputs=missing_inputs,
        )
        if pred_row is None:
            return None

        latest_window = recent_windows[-1]

        predicted_components = self._build_predicted_components(
            pred_row=pred_row,
            target_columns=target_columns,
            latest_window=latest_window,
        )

        latest_window = recent_windows[-1]
        expected_input_count = len(recent_windows) * len(input_columns)
        confidence = self._estimate_prediction_confidence(
            missing_inputs=missing_inputs,
            expected_input_count=expected_input_count,
        )

        predicted_features = self._build_predicted_features(
            latest_window=latest_window,
            predicted_components=predicted_components,
            forecast_id=forecast_id,
            horizon_seconds=horizon_seconds,
            confidence=confidence,
        )

        infer_elapsed_ms = (time.perf_counter() - infer_started_at) * 1000.0
        self._logger.system(
            "forecasting_tool_inference_timing",
            {
                "inference_ms": round(infer_elapsed_ms, 3),
                "confidence": round(predicted_features.confidence, 3),
                "target_columns": target_columns,
            },
            level="DEBUG",
        )

        return predicted_features

    # -------------------------------------------------------------------------
    # Internal: misc
    # -------------------------------------------------------------------------

    def _estimate_uncertainty(
        self,
        prediction: PredictedFeatures,
    ) -> dict:
        return {}

    def _should_update_prediction(self) -> bool:
        update_rate = float(self._config.update_rate_hz)
        if update_rate <= 0:
            return True

        min_interval = 1.0 / update_rate
        now = time.time()
        if now - self._last_prediction_time < min_interval:
            return False

        self._last_prediction_time = now
        return True
