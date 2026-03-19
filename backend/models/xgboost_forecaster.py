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
        self._models: Dict[str, xgb.XGBRegressor] = {}
        self._metadata: Dict[str, Any] = {}
        self._history_size: int = 5
        self._feature_names: List[str] = []
        self._target_columns: List[str] = TARGET_COLUMNS.copy()
        self._target_scaler_mean: Optional[np.ndarray] = None
        self._target_scaler_std: Optional[np.ndarray] = None

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

        self._model = None
        self._models = {}
        self._metadata = {}
        self._target_scaler_mean = None
        self._target_scaler_std = None
        self._target_columns = TARGET_COLUMNS.copy()

        model_path = self._resolve_json_pointer(model_path)
        metadata_path = self._resolve_metadata_path(model_path)

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self._metadata = json.load(f)
                self._history_size = self._metadata.get('history_window_size', 5)
                self._feature_names = self._metadata.get('feature_names', [])
                metadata_targets = self._metadata.get("target_columns")
                if isinstance(metadata_targets, list) and metadata_targets:
                    self._target_columns = [str(col) for col in metadata_targets]
                target_scaler = self._metadata.get("target_scaler")
                if isinstance(target_scaler, dict):
                    mean = target_scaler.get("mean")
                    std = target_scaler.get("std")
                    if (
                        isinstance(mean, list)
                        and isinstance(std, list)
                        and len(mean) == len(self._target_columns)
                        and len(std) == len(self._target_columns)
                    ):
                        self._target_scaler_mean = np.asarray(mean, dtype=float)
                        safe_std = np.asarray(std, dtype=float)
                        safe_std = np.where(np.abs(safe_std) < 1e-8, 1.0, safe_std)
                        self._target_scaler_std = safe_std
            except Exception as e:
                print(f"Warning: Could not parse metadata file {metadata_path}: {e}")

        model_files = self._metadata.get("model_files")
        if isinstance(model_files, dict) and model_files:
            loaded_models: Dict[str, xgb.XGBRegressor] = {}
            for target in self._target_columns:
                model_file = model_files.get(target)
                if not isinstance(model_file, str) or not model_file:
                    raise ValueError(f"Missing model file entry for target '{target}'")

                target_model_path = Path(model_file)
                if not target_model_path.is_absolute():
                    target_model_path = (model_path.parent / target_model_path).resolve()
                target_model_path = self._resolve_json_pointer(target_model_path)
                if not target_model_path.exists():
                    raise FileNotFoundError(
                        f"Model file for target '{target}' not found: {target_model_path}"
                    )

                target_model = xgb.XGBRegressor()
                target_model.load_model(str(target_model_path))
                loaded_models[target] = target_model

            self._models = loaded_models
            self._model = self._models[self._target_columns[0]]
            print(f"Loaded ensemble model ({len(self._models)} targets) from: {model_path}")
            return True

        # Backward-compatible single-model loading.
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_path))
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

    def _resolve_metadata_path(self, model_path: Path) -> Path:
        """
        Resolve metadata path for both legacy single-model and new ensemble names.
        """
        if model_path.stem == "latest":
            return self._resolve_json_pointer(model_path.with_name("latest_metadata.json"))

        candidates = [model_path.with_name(f"{model_path.stem}_metadata.json")]

        # Support suffix stripping for both contributor-space and legacy raw-target model names.
        known_suffixes = list(dict.fromkeys(
            TARGET_COLUMNS + [
                "pupil_ipa",
                "fixation_mean_duration_ms",
                "saccade_mean_velocity",
                "saccade_velocity_std",
                "ipi_value",
            ]
        ))

        for target in known_suffixes:
            suffix = f"_{target}"
            if model_path.stem.endswith(suffix):
                base_stem = model_path.stem[: -len(suffix)]
                candidates.insert(0, model_path.with_name(f"{base_stem}_metadata.json"))
                break

        for candidate in candidates:
            resolved = self._resolve_json_pointer(candidate)
            if resolved.exists():
                return resolved

        return self._resolve_json_pointer(candidates[0])

    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return bool(self._models) or self._model is not None

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
        if self._models:
            prediction = np.array(
                [self._models[target].predict(X)[0] for target in self._target_columns],
                dtype=float,
            )
        elif self._model is not None:
            prediction = self._model.predict(X)[0]
        else:
            print("Model not loaded")
            return None

        # Multi-target model: return expected target component mapping.
        if (
            isinstance(prediction, np.ndarray)
            and prediction.ndim == 1
            and len(prediction) == len(self._target_columns)
        ):
            if self._target_scaler_mean is not None and self._target_scaler_std is not None:
                prediction = prediction * self._target_scaler_std + self._target_scaler_mean

            prediction_map = {
                key: float(prediction[idx])
                for idx, key in enumerate(self._target_columns)
            }

            # If model already predicts contributor-space targets, return directly.
            if all(col in prediction_map for col in TARGET_COLUMNS):
                return {col: float(prediction_map[col]) for col in TARGET_COLUMNS}

            # Legacy compatibility: raw-target models are converted to contributor space.
            legacy_contribs = compute_contributor_features(prediction_map)
            return {
                key: float(legacy_contribs[key])
                for key in TARGET_COLUMNS
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
            'model_type': self._metadata.get('model_type'),
            'model_count': len(self._models) if self._models else (1 if self._model is not None else 0),
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
