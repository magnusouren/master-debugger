"""
Forecasting Tool (Proactive mode only)

Position: Between Signal Processing and Reactive Tool
Input: Window-based features (e.g., 2–10 Hz)
Output: Predicted window-based features (same format)
Configuration: Prediction horizon (seconds into the future), Update rate

This component predicts future feature values based on recent observations.
Predictions are produced using a trained model based on historical or 
previously collected eye-tracking data. The output format is identical 
to the Signal Processing output to ensure compatibility with downstream logic.
"""
from typing import Optional, List, Callable
from collections import deque
import time

from backend.services.logger_service import LoggerService
from backend.types import (
    WindowFeatures,
    PredictedFeatures,
    ForecastingConfig,
)
from backend.models.xgboost_forecaster import XGBoostForecaster
from backend.models.forecast_feature_schema import TARGET_COLUMNS


class ForecastingTool:
    """
    Predicts future eye-tracking features for proactive intervention.
    """
    
    def __init__(
        self,
        config: Optional[ForecastingConfig] = None,
        logger: Optional[LoggerService] = None,
    ):
        """
        Initialize the Forecasting Tool.
        
        Args:
            config: Configuration for forecasting parameters.
            logger: Logger instance for recording events.
        """
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

        # Use provided logger or create fallback
        if logger is None:
            from backend.services.logger_service import get_logger
            logger = get_logger()
        self._logger = logger

        # Auto-load model if path specified in config
        if self._config.model_path:
            self.load_model(self._config.model_path)
        else:
            self._logger.system(
                "forecasting_tool_no_model_configured",
                {},
                level="WARNING"
            )
    
    def configure(self, config: ForecastingConfig) -> None:
        """
        Update forecasting configuration.
        
        Args:
            config: New configuration to apply.
        """
        self._config = config
    
    def enable(self) -> None:
        """Enable the forecasting tool (for proactive mode)."""
        # Ensure a model is available when proactive mode starts.
        if not self._forecaster or not self._forecaster.is_loaded():
            self.load_model(self._config.model_path)
        self._is_enabled = True
    
    def disable(self) -> None:
        """Disable the forecasting tool (for reactive mode)."""
        self._is_enabled = False
    
    def is_enabled(self) -> bool:
        """
        Check if forecasting is currently enabled.
        
        Returns:
            True if forecasting is active.
        """
        return self._is_enabled
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained forecasting model.

        Args:
            model_path: Path to the model file. If None, loads latest model.

        Returns:
            True if model loaded successfully.
        """
        from pathlib import Path

        try:
            self._forecaster = XGBoostForecaster()
            path = Path(model_path) if model_path else None
            success = self._forecaster.load_model(path)

            if success:
                self._logger.system(
                    "forecasting_tool_model_loaded",
                    {"model_path": str(model_path), "info": self._forecaster.get_model_info()},
                )
            else:
                self._logger.system(
                    "forecasting_tool_model_load_failed",
                    {"model_path": str(model_path)},
                    level="WARNING"
                )

            return success
        except Exception as e:
            self._logger.system(
                "forecasting_tool_model_load_error",
                {"model_path": str(model_path), "error": str(e)},
                level="ERROR"
            )
            return False
    
    def unload_model(self) -> None:
        """Unload the current model and free resources."""
        self._forecaster = None
        self._logger.system("forecasting_tool_model_unloaded", {})
    
    def add_features(self, features: WindowFeatures) -> None:
        # 1. Add new features to history buffer
        self._feature_history.append(features)

        # 2. Calculate cutoff time for history retention
        cutoff_time = features.window_end - self._config.history_window_seconds

        # 3. Remove all windows older than cutoff
        while (
            self._feature_history
            and self._feature_history[0].window_end < cutoff_time
        ):
            self._feature_history.popleft()

        # 4. Trigger prediction if allowed
        if self._is_enabled and self._should_update_prediction():
            prediction = self.predict()
            if (
                prediction
                and prediction.confidence >= self._config.min_confidence_threshold
            ):
                for callback in self._output_callbacks:
                    try:
                        callback(prediction)
                    except Exception as e:
                        self._logger.system(
                            "forecasting_tool_callback_error",
                            {"error": str(e)},
                            level="ERROR"
                        )
    
    def predict(self) -> Optional[PredictedFeatures]:
        """
        Generate a prediction for future features.
        
        Returns:
            Predicted features or None if prediction not possible.
        """

        if not self._config.prediction_horizon_seconds:
            self._logger.system(
                "forecasting_tool_no_horizon_configured",
                {},
                level="WARNING"
            )
            return None

        return self.predict_at_horizon(self._config.prediction_horizon_seconds)
    
    def predict_at_horizon(
        self, horizon_seconds: float
    ) -> Optional[PredictedFeatures]:
        """
        Generate a prediction for a specific time horizon.
        
        Args:
            horizon_seconds: How far into the future to predict.
            
        Returns:
            Predicted features or None if prediction not possible.
        """

        started_at = time.perf_counter()
        forecast_id = self._next_forecast_id()
        window_feature = self._prepare_input_sequence()

        if not window_feature:
            return None
        
        prediction = self._run_model_inference(window_feature, forecast_id, horizon_seconds)

        if prediction is not None:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self._logger.system(
                "forecasting_tool_prediction_latency",
                {
                    "latency_ms": round(elapsed_ms, 3),
                    "history_windows": len(window_feature),
                    "horizon_seconds": horizon_seconds,
                },
                level="DEBUG",
            )

        return prediction
    
    def get_prediction_confidence(self) -> float:
        """
        Get the confidence of the latest prediction.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Get confidence from the most recent prediction
        if not self._forecaster or not self._forecaster.is_loaded():
            return 0.0

        input_sequence = self._prepare_input_sequence()
        if not input_sequence:
            return 0.0

        _, confidence = self._forecaster.predict_with_confidence(input_sequence)
        return confidence
    
    def register_output_callback(
        self, callback: Callable[[PredictedFeatures], None]
    ) -> None:
        """
        Register a callback to receive predicted features.
        
        Args:
            callback: Function to call with predictions.
        """
        self._output_callbacks.append(callback)
    
    def unregister_output_callback(
        self, callback: Callable[[PredictedFeatures], None]
    ) -> None:
        """
        Unregister a previously registered callback.
        
        Args:
            callback: The callback function to remove.
        """
        if callback in self._output_callbacks:
            self._output_callbacks.remove(callback)
    
    def reset(self) -> None:
        """Reset internal state and history buffer."""
        self._feature_history.clear()
        self._last_prediction_time = 0.0
        self._last_warmup_status = None
        self._forecast_counter = 0
        self._pred_window_counter = 0

    def _next_forecast_id(self) -> str:
        self._forecast_counter += 1
        return f"{self._id_prefix}-f{self._forecast_counter:06d}"

    def _next_pred_window_id(self, forecast_id: str) -> str:
        self._pred_window_counter += 1
        return f"{forecast_id}-w{self._pred_window_counter:04d}"

    def _compute_required_history(self) -> int:
        if self._forecaster and self._forecaster.is_loaded():
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
        buffer_fill_ratio = 0.0 if required_windows <= 0 else min(1.0, buffer_windows / required_windows)

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
    
    # --- Internal methods ---
    
    def _prepare_input_sequence(self) -> Optional[List[WindowFeatures]]:
        """
        Prepare the input sequence for the prediction model.

        Returns:
            Prepared feature sequence or None if insufficient data.
        """
        required_history = self._compute_required_history()
        can_predict = len(self._feature_history) >= required_history
        self._log_warmup_status(can_predict=can_predict, required_windows=required_history)

        if not can_predict:
            return None

        return list(self._feature_history)
    
    def _run_model_inference(
        self,
        input_sequence: List[WindowFeatures],
        forecast_id: str,
        horizon_seconds: float,
    ) -> Optional[PredictedFeatures]:
        """
        Run the forecasting model on input sequence.

        Args:
            input_sequence: Prepared feature sequence.

        Returns:
            Model prediction output, or None if prediction fails.
        """
        if not self._forecaster or not self._forecaster.is_loaded():
            self._logger.system(
                "forecasting_tool_no_model",
                {"message": "No model loaded for inference"},
                level="WARNING"
            )
            return None

        # Get prediction from XGBoost model
        infer_started_at = time.perf_counter()
        predicted_components, confidence = self._forecaster.predict_with_confidence(input_sequence)
        infer_elapsed_ms = (time.perf_counter() - infer_started_at) * 1000.0

        if predicted_components is None:
            return None

        missing_targets = [k for k in TARGET_COLUMNS if k not in predicted_components]
        if missing_targets:
            self._logger.system(
                "forecasting_tool_prediction_missing_targets",
                {"missing_targets": missing_targets},
                level="WARNING",
            )
            return None

        # Get the latest window for timing info
        latest_window = input_sequence[-1]
        window_duration = latest_window.window_end - latest_window.window_start

        # Create PredictedFeatures with predicted contributor values.
        # ReactiveTool computes final score (incl. optional baseline normalization).
        predicted_features = PredictedFeatures(
            prediction_timestamp=latest_window.window_end,
            target_window_start=latest_window.window_end + horizon_seconds,
            target_window_end=latest_window.window_end + horizon_seconds + window_duration,
            horizon_seconds=horizon_seconds,
            forecast_id=forecast_id,
            window_id=self._next_pred_window_id(forecast_id),
            features=predicted_components,
            enabled_metrics=latest_window.enabled_metrics.copy(),
            confidence=confidence,
            uncertainty={},
        )

        self._logger.system(
            "forecasting_tool_inference_timing",
            {
                "inference_ms": round(infer_elapsed_ms, 3),
                "confidence": round(confidence, 3),
            },
            level="DEBUG",
        )

        return predicted_features
    
    def _estimate_uncertainty(
        self, prediction: PredictedFeatures
    ) -> dict:
        """
        Estimate uncertainty for each predicted feature.
        
        Args:
            prediction: The model's prediction.
            
        Returns:
            Dictionary mapping feature names to uncertainty values.
        """

        return {}  # TODO: Implement uncertainty estimation
    
    def _should_update_prediction(self) -> bool:
        """
        Check if a new prediction should be generated based on update rate.
        
        Returns:
            True if prediction should be updated.
        """
        update_rate = float(self._config.update_rate_hz)
        if update_rate <= 0:
            return True

        min_interval = 1.0 / update_rate
        now = time.time()
        if now - self._last_prediction_time < min_interval:
            return False

        self._last_prediction_time = now
        return True
