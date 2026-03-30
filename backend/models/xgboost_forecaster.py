"""
XGBoost Forecaster - runtime inference wrapper for WindowFeatures forecasting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DEFAULT_INPUT_COLUMNS = [
    "pupil_ipa",
    # "fixation_mean_duration_ms",
]

DEFAULT_TARGET_COLUMNS = [
    "pupil_ipa",
    # "fixation_mean_duration_ms",
]


class XGBoostForecaster:
    """
    Runtime wrapper for an XGBoost forecasting model trained on WindowFeatures.
    """

    def __init__(self, model_path: Optional[Path] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self._model: Optional[xgb.XGBRegressor] = None
        self._metadata: Dict[str, Any] = {}
        self._history_size: int = 5
        self._feature_names: List[str] = []
        self._input_columns: List[str] = list(DEFAULT_INPUT_COLUMNS)
        self._target_columns: List[str] = list(DEFAULT_TARGET_COLUMNS)

        if model_path is not None:
            self.load_model(model_path)

    # -------------------------------------------------------------------------
    # Loading / metadata
    # -------------------------------------------------------------------------

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        if model_path is None:
            base_dir = Path(__file__).parent
            model_path = base_dir / "trained" / "latest.json"

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return False

        model_path = self._resolve_json_pointer(model_path)

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_path))

        metadata_path = self._resolve_metadata_path(model_path)
        self._load_metadata(metadata_path)

        print(f"Loaded model: {model_path}")
        return True

    def _resolve_metadata_path(self, model_path: Path) -> Path:
        if model_path.stem == "latest":
            metadata_path = model_path.with_name("latest_metadata.json")
        else:
            metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
        return self._resolve_json_pointer(metadata_path)

    def _load_metadata(self, metadata_path: Path) -> None:
        self._metadata = {}

        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse metadata file {metadata_path}: {e}")
            return

        self._history_size = int(self._metadata.get("history_window_size", 5))
        self._feature_names = list(self._metadata.get("feature_names", []))
        self._input_columns = list(self._metadata.get("input_columns", DEFAULT_INPUT_COLUMNS))
        self._target_columns = list(self._metadata.get("target_columns", DEFAULT_TARGET_COLUMNS))

    def _resolve_json_pointer(self, path: Path) -> Path:
        try:
            if not path.exists() or path.suffix != ".json":
                return path

            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text and not text.startswith("{"):
                candidate = (path.parent / text).resolve()
                if candidate.exists() and candidate.suffix == ".json":
                    return candidate
        except Exception:
            pass

        return path

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def history_size(self) -> int:
        return self._history_size

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def input_columns(self) -> List[str]:
        return list(self._input_columns)

    @property
    def target_columns(self) -> List[str]:
        return list(self._target_columns)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict_windows(
        self,
        history: List[Any],
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """
        Predict target columns from recent WindowFeatures history.

        Returns:
            (prediction_dict, confidence)
        """
        if not self.is_loaded():
            print("Model not loaded")
            return None, 0.0

        recent = self._select_recent_windows(history)
        if recent is None:
            return None, 0.0

        flat_features, missing_inputs = self._flatten_windows(recent)

        try:
            pred_arr = self._predict_array(flat_features)
        except ValueError as e:
            print(f"Inference failed: {e}")
            return None, 0.0

        pred_row = self._extract_prediction_row(pred_arr)
        if pred_row is None:
            return None, 0.0

        prediction = self._prediction_row_to_dict(pred_row)
        confidence = self._estimate_confidence(
            missing_inputs=missing_inputs,
            expected_input_count=len(recent) * len(self._input_columns),
        )
        return prediction, confidence

    def _select_recent_windows(self, history: List[Any]) -> Optional[List[Any]]:
        if len(history) < self._history_size:
            print(f"Not enough history: {len(history)} < {self._history_size}")
            return None

        return list(history)[-self._history_size:]

    def _flatten_windows(self, windows: List[Any]) -> Tuple[List[float], List[str]]:
        flat_features: List[float] = []
        missing_inputs: List[str] = []

        for window in windows:
            feature_dict = window.features if hasattr(window, "features") else window
            feature_dict = feature_dict or {}

            for col in self._input_columns:
                raw_val = feature_dict.get(col) if isinstance(feature_dict, dict) else None
                if raw_val is None:
                    missing_inputs.append(col)
                flat_features.append(self._safe_float(raw_val))

        return flat_features, missing_inputs

    def _predict_array(self, flat_features: List[float]) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not loaded")

        X = np.array([flat_features], dtype=np.float32)

        try:
            prediction = self._model.predict(X)
        except ValueError as e:
            expected_features = getattr(
                self._model,
                "n_features_in_",
                len(self._input_columns) * self._history_size,
            )
            raise ValueError(
                "Feature shape mismatch during forecasting inference: "
                f"expected_features={expected_features}, got_features={X.shape[1]}, "
                f"history_size={self._history_size}, input_columns={self._input_columns}"
            ) from e

        return np.asarray(prediction)

    def _extract_prediction_row(self, pred_arr: np.ndarray) -> Optional[np.ndarray]:
        if pred_arr.ndim == 2 and pred_arr.shape[0] >= 1:
            pred_row = pred_arr[0]
        elif pred_arr.ndim == 1:
            pred_row = pred_arr
        else:
            return None

        if len(pred_row) < len(self._target_columns):
            return None

        return pred_row

    def _prediction_row_to_dict(self, pred_row: np.ndarray) -> Dict[str, float]:
        return {
            col: float(pred_row[idx])
            for idx, col in enumerate(self._target_columns)
        }

    def _estimate_confidence(
        self,
        missing_inputs: List[str],
        expected_input_count: int,
    ) -> float:
        if expected_input_count <= 0:
            return 0.0

        missing_ratio = len(missing_inputs) / expected_input_count
        confidence = 1.0 - missing_ratio
        return float(max(0.0, min(1.0, confidence)))

    def _safe_float(self, value: Any) -> float:
        try:
            if value is None:
                return float("nan")

            arr = np.asarray(value)
            if arr.size == 0:
                return float("nan")
            if arr.ndim == 0:
                return float(arr)

            return float(np.nanmean(arr))
        except Exception:
            return float("nan")

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        if not self._metadata:
            return {"loaded": self.is_loaded()}

        return {
            "loaded": self.is_loaded(),
            "version": self._metadata.get("version"),
            "history_window_size": self._history_size,
            "prediction_horizon": self._metadata.get("prediction_horizon"),
            "input_columns": self._input_columns,
            "target_columns": self._target_columns,
            "metrics": self._metadata.get("metrics", {}),
        }


_default_forecaster: Optional[XGBoostForecaster] = None


def get_forecaster() -> XGBoostForecaster:
    global _default_forecaster
    if _default_forecaster is None:
        _default_forecaster = XGBoostForecaster()
        try:
            _default_forecaster.load_model()
        except Exception as e:
            print(f"Warning: Could not load default model: {e}")
    return _default_forecaster