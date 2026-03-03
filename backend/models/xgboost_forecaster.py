"""
XGBoost Forecaster - Inference wrapper for cognitive load prediction.

This class is used by ForecastingTool to make predictions at runtime.
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from backend.models.forecast_feature_schema import FEATURE_COLUMNS, compute_contributor_features

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

        # Load model
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_path))

        # Load metadata
        metadata_path = model_path.with_name(
            model_path.stem.replace('.json', '') + "_metadata.json"
        )
        if model_path.stem == "latest":
            metadata_path = model_path.with_name("latest_metadata.json")

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self._metadata = json.load(f)
            self._history_size = self._metadata.get('history_window_size', 5)
            self._feature_names = self._metadata.get('feature_names', [])

        print(f"Loaded model: {model_path}")
        return True

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

    def predict(self, history: List[Dict[str, Any]]) -> Optional[float]:
        """
        Predict future cognitive load score.

        Args:
            history: List of WindowFeatures dicts (most recent last)

        Returns:
            Predicted cognitive load score (0-1), or None if prediction fails
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

        # Clip to valid range
        return float(max(0.0, min(1.0, prediction)))

    def predict_with_confidence(
        self, history: List[Dict[str, Any]]
    ) -> tuple[Optional[float], float]:
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
            for col in FEATURE_COLUMNS:
                val = window_dict.get(col)
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
