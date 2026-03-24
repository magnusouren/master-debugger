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
        self._input_columns: List[str] = []
        self._target_columns: List[str] = list(TARGET_COLUMNS)

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
        metadata_path = self._resolve_metadata_path(model_path)
        if metadata_path is not None:
            loaded_metadata = self._load_metadata(metadata_path)
            if loaded_metadata:
                self._metadata = loaded_metadata

        self._history_size = int(self._metadata.get("history_window_size", self._history_size))
        self._feature_names = list(self._metadata.get("feature_names", []))
        self._input_columns = list(self._metadata.get("input_columns", []))
        self._target_columns = list(self._metadata.get("target_columns", TARGET_COLUMNS))

        # If metadata is incomplete, infer history size from model dimensionality.
        inferred_history = self._infer_history_size_from_model()
        if inferred_history is not None:
            self._history_size = inferred_history

        print(f"Loaded model: {model_path}")
        return True

    def _resolve_metadata_path(self, model_path: Path) -> Optional[Path]:
        """Resolve metadata path for a model, including tolerant fallback for `latest` files."""
        candidates: List[Path] = []

        if model_path.stem == "latest":
            candidates.append(model_path.with_name("latest_metadata.json"))
            candidates.extend(sorted(model_path.parent.glob("latest_metadata*.json")))
        else:
            candidates.append(model_path.with_name(f"{model_path.stem}_metadata.json"))

        for candidate in candidates:
            resolved = self._resolve_json_pointer(candidate)
            if resolved.exists():
                return resolved
        return None

    def _load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """Load metadata with tolerant handling for non-standard JSON constants."""
        try:
            text = metadata_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read metadata file {metadata_path}: {e}")
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Some metadata snapshots contain NaN; coerce such constants to None.
                return json.loads(text, parse_constant=lambda _v: None)
            except Exception as e:
                print(f"Warning: Could not parse metadata file {metadata_path}: {e}")
                return {}

    def _infer_history_size_from_model(self) -> Optional[int]:
        """Infer history size from model feature count and per-window feature count."""
        if self._model is None:
            return None

        model_feature_count = getattr(self._model, "n_features_in_", None)
        if model_feature_count is None:
            model_feature_count = self._metadata.get("n_features")
        if model_feature_count is None:
            return None

        try:
            model_feature_count = int(model_feature_count)
        except Exception:
            return None

        per_window_feature_count = len(self._input_columns) if self._input_columns else len(FEATURE_COLUMNS)
        if per_window_feature_count <= 0:
            return None

        if model_feature_count % per_window_feature_count != 0:
            return None

        return max(1, model_feature_count // per_window_feature_count)

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
        if self._input_columns:
            values: List[float] = []
            for col in self._input_columns:
                raw = window_features.get(col)
                if raw is None:
                    values.append(0.0)
                    continue
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    values.append(0.0)
            return values

        contribs = compute_contributor_features(window_features)
        return [float(contribs[col]) for col in FEATURE_COLUMNS]

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

        # Extract and flatten features
        features = []
        for window in recent:
            # Handle both WindowFeatures objects and dicts
            if hasattr(window, 'features'):
                window_dict = window.features
            else:
                window_dict = window

            features.extend(self.extract_features_from_window(window_dict))

        # Predict
        X = np.array([features])
        prediction = self._model.predict(X)[0]

        expected_targets = self._target_columns or TARGET_COLUMNS

        # Multi-target model: map values using target columns from metadata.
        if isinstance(prediction, np.ndarray) and prediction.ndim == 1 and len(prediction) == len(expected_targets):
            return {
                key: float(prediction[idx])
                for idx, key in enumerate(expected_targets)
            }

        # Backward-compatible fallback for older single-target models.
        if np.isscalar(prediction):
            scalar = float(prediction)
            return {"predicted_cognitive_load": max(0.0, min(1.0, scalar))}

        # Last resort: map any vector output positionally.
        if isinstance(prediction, np.ndarray) and prediction.ndim == 1:
            names = expected_targets if len(expected_targets) == len(prediction) else [f"target_{i}" for i in range(len(prediction))]
            return {name: float(prediction[idx]) for idx, name in enumerate(names)}

        return None

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
        feature_columns = self._input_columns if self._input_columns else FEATURE_COLUMNS
        for window in recent:
            window_dict = window.features if hasattr(window, 'features') else window
            if self._input_columns:
                for col in self._input_columns:
                    val = window_dict.get(col)
                    if val is None:
                        missing_count += 1
                        continue
                    try:
                        if np.isnan(float(val)):
                            missing_count += 1
                    except (TypeError, ValueError):
                        missing_count += 1
            else:
                contribs = compute_contributor_features(window_dict)
                for col in FEATURE_COLUMNS:
                    val = contribs.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        missing_count += 1

        total_features = len(feature_columns) * self._history_size
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
            'input_columns': self._input_columns,
            'target_columns': self._target_columns,
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
