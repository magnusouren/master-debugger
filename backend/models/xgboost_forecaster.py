"""
XGBoost Forecaster - Inference wrapper for cognitive load prediction.

This class is used by ForecastingTool to make predictions at runtime.
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from backend.models.forecast_feature_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    compute_contributor_features,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostForecaster:
    """
    Wrapper for XGBoost cognitive load forecasting model.

    Handles loading the model and making predictions from WindowFeatures.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the forecaster.

        Args:
            model_path: Path to model file. If None, tries to load latest model.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self._model: Optional[xgb.XGBRegressor] = None
        self._metadata: Dict[str, Any] = {}
        self._history_size: int = 5
        self._feature_names: List[str] = []

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load a trained model.

        Args:
            model_path: Path to model file. If None, loads latest model.

        Returns:
            True if model loaded successfully.
        """
        if model_path is None:
            # Try to load latest model
            base_dir = Path(__file__).parent
            model_path = base_dir / "trained" / "latest.json"

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return False

        model_path = self._resolve_json_pointer(model_path)

        # Load model
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_path))

        # Load metadata
        metadata_path = model_path.with_name(
            model_path.stem.replace('.json', '') + "_metadata.json"
        )
        if model_path.stem == "latest":
            metadata_path = model_path.with_name("latest_metadata.json")
        metadata_path = self._resolve_json_pointer(metadata_path)

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self._metadata = json.load(f)
                self._history_size = self._metadata.get('history_window_size', 5)
                self._feature_names = self._metadata.get('feature_names', [])
            except Exception as e:
                print(f"Warning: Could not parse metadata file {metadata_path}: {e}")

        print(f"Loaded model: {model_path}")
        return True

    def _resolve_json_pointer(self, path: Path) -> Path:
        """
        Resolve git symlink placeholder files on systems without symlink support.

        If a .json file contains plain text instead of JSON, treat that text as
        a relative target filename and return the resolved path when it exists.
        """
        try:
            if not path.exists() or path.suffix != ".json":
                return path

            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text and not text.startswith("{"):
                candidate = (path.parent / text).resolve()
                if candidate.exists() and candidate.suffix == ".json":
                    return candidate
        except Exception:
            # Best-effort only; caller continues with original path.
            pass
        return path

    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model is not None

    @property
    def history_size(self) -> int:
        """Number of historical windows required for prediction."""
        return self._history_size

    @property
    def feature_names(self) -> List[str]:
        """Names of input features."""
        return self._feature_names

    def extract_features_from_window(self, window_features: Dict[str, Any]) -> List[float]:
        """
        Extract feature values from a WindowFeatures dict.

        Args:
            window_features: Dictionary of features from signal processing

        Returns:
            List of feature values in the expected order
        """
        contribs = compute_contributor_features(window_features)
        features = [float(contribs[col]) for col in FEATURE_COLUMNS]

        return features

    def _safe_float(self, value: Any) -> float:
        """Best-effort float conversion; handles arrays via nanmean."""
        try:
            if value is None:
                return float('nan')

            arr = np.asarray(value)
            if arr.size == 0:
                return float('nan')
            if arr.ndim == 0:
                return float(arr)

            # For vectors/matrices, fall back to mean to avoid scalar cast errors
            return float(np.nanmean(arr))
        except Exception:
            return float('nan')

    def _extract_features_by_model_names(
        self,
        history: List[Dict[str, Any]],
        model_feature_names: List[str],
    ) -> List[float]:
        """
        Build the feature vector following the model's recorded feature names.

        Supports names like "t-59_pupil_ipa" where t-0 is the most recent
        window. Falls back to NaN when history is insufficient.
        """
        features: List[float] = []
        for name in model_feature_names:
            value = float('nan')
            idx = None
            metric = name

            # Parse time-offset format: t-<k>_<metric>
            if name.startswith("t-") and "_" in name[2:]:
                try:
                    offset_part, metric = name.split("_", 1)
                    k = int(offset_part.replace("t-", ""))
                    idx = len(history) - 1 - k  # t-0 is latest
                except ValueError:
                    idx = None

            if idx is not None and 0 <= idx < len(history):
                window = history[idx]
                window_dict = window.features if hasattr(window, "features") else window
                raw_val = None
                if isinstance(window_dict, dict):
                    raw_val = window_dict.get(metric)
                value = self._safe_float(raw_val)

            features.append(value)

        return features

    def predict(self, history: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """
        Predict future target component values.

        Args:
            history: List of WindowFeatures dicts (most recent last)

        Returns:
            Predicted target component values, or None if prediction fails
        """
        if not self.is_loaded():
            print("Model not loaded")
            return None

        if len(history) < self._history_size:
            print(f"Not enough history: {len(history)} < {self._history_size}")
            return None

        # Take the most recent windows
        recent = history[-self._history_size:]

        # Extract and flatten features using model names when available
        model_feature_names = list(getattr(self._model, "feature_names_in_", [])) or list(self._feature_names)

        if model_feature_names:
            features = self._extract_features_by_model_names(recent, model_feature_names)
        else:
            features = []
            for window in recent:
                window_dict = window.features if hasattr(window, 'features') else window
                features.extend(self.extract_features_from_window(window_dict))

        # Predict
        X = np.array([features])

        # Capture the most recent window keys for debugging shape mismatches
        last_window = recent[-1]
        last_window_dict = last_window.features if hasattr(last_window, "features") else last_window
        last_window_keys = (
            sorted(list(last_window_dict.keys()))
            if isinstance(last_window_dict, dict)
            else "unavailable"
        )

        try:
            prediction = self._model.predict(X)[0]
        except ValueError as e:
            expected_features = getattr(
                self._model,
                "n_features_in_",
                len(model_feature_names) or len(self._feature_names) or len(FEATURE_COLUMNS) * self._history_size,
            )
            raise ValueError(
                "Feature shape mismatch during forecasting inference: "
                f"expected_features={expected_features}, "
                f"got_features={X.shape[1]}, "
                f"history_size={self._history_size}, "
                f"features_per_window={len(FEATURE_COLUMNS)}, "
                f"metadata_feature_names={len(self._feature_names)}, "
                f"model_feature_names={getattr(self._model, 'feature_names_in_', None)}, "
                f"last_window_keys={last_window_keys}, "
                f"metadata_version={self._metadata.get('version') if self._metadata else None}"
            ) from e

        # Multi-target model: return expected target component mapping.
        if isinstance(prediction, np.ndarray) and prediction.ndim == 1 and len(prediction) == len(TARGET_COLUMNS):
            return {
                key: float(prediction[idx])
                for idx, key in enumerate(TARGET_COLUMNS)
            }

        # Backward-compatible fallback for older single-target models.
        scalar = float(prediction)
        return {"predicted_cognitive_load": max(0.0, min(1.0, scalar))}

    def predict_with_confidence(
        self, history: List[Dict[str, Any]]
    ) -> tuple[Optional[Dict[str, float]], float]:
        """
        Predict with confidence estimate.

        Returns:
            (prediction, confidence) tuple
        """
        prediction = self.predict(history)

        if prediction is None:
            return None, 0.0

        # Simple confidence based on data quality
        # In production, could use prediction intervals or ensemble variance
        confidence = 0.7  # Base confidence

        # Reduce confidence if missing data
        recent = history[-self._history_size:]
        missing_count = 0
        for window in recent:
            window_dict = window.features if hasattr(window, 'features') else window
            contribs = compute_contributor_features(window_dict)
            for col in FEATURE_COLUMNS:
                val = contribs.get(col)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    missing_count += 1

        total_features = len(FEATURE_COLUMNS) * self._history_size
        missing_ratio = missing_count / total_features
        confidence = confidence * (1 - missing_ratio * 0.5)

        return prediction, confidence

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        if not self._metadata:
            return {'loaded': self.is_loaded()}

        return {
            'loaded': self.is_loaded(),
            'version': self._metadata.get('version'),
            'history_window_size': self._history_size,
            'prediction_horizon': self._metadata.get('prediction_horizon'),
            'metrics': self._metadata.get('metrics', {}),
        }


# Singleton instance for easy access
_default_forecaster: Optional[XGBoostForecaster] = None


def get_forecaster() -> XGBoostForecaster:
    """Get or create the default forecaster instance."""
    global _default_forecaster
    if _default_forecaster is None:
        _default_forecaster = XGBoostForecaster()
        try:
            _default_forecaster.load_model()
        except Exception as e:
            print(f"Warning: Could not load default model: {e}")
    return _default_forecaster
