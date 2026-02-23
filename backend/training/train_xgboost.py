"""
Train XGBoost model for cognitive load forecasting.

Usage:
    python -m backend.training.train_xgboost
    python -m backend.training.train_xgboost --config backend/config.yaml
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    exit(1)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.types import TrainingConfig, SystemConfig
from backend.training.dataset import prepare_dataset


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    config: TrainingConfig,
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor with early stopping.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names
        config: TrainingConfig with hyperparameters

    Returns:
        Trained XGBoost model
    """
    print("\n--- Training XGBoost ---")

    model = xgb.XGBRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=config.random_state,
        n_jobs=-1,
        early_stopping_rounds=config.early_stopping_rounds,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best validation score: {model.best_score:.4f}")

    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        X_test, y_test: Test data
        threshold: Threshold for binary classification metrics

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n--- Evaluation on Test Set ---")

    y_pred = model.predict(X_test)

    # Regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # Binary classification metrics (above/below threshold)
    y_test_binary = (y_test >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    accuracy = (y_test_binary == y_pred_binary).mean()
    true_positives = ((y_test_binary == 1) & (y_pred_binary == 1)).sum()
    false_positives = ((y_test_binary == 0) & (y_pred_binary == 1)).sum()
    false_negatives = ((y_test_binary == 1) & (y_pred_binary == 0)).sum()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nBinary classification (threshold={threshold}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def get_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: list,
    top_n: int = 20
) -> dict:
    """Get top feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n--- Top {top_n} Feature Importances ---")
    importance_dict = {}
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")
        importance_dict[feature_names[i]] = float(importances[i])

    return importance_dict


def save_model(
    model: xgb.XGBRegressor,
    metrics: dict,
    feature_names: list,
    output_dir: Path,
    config: TrainingConfig,
) -> Path:
    """
    Save trained model and metadata.

    Returns path to saved model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate version string
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgb_forecaster_{version}"

    # Save model
    model_path = output_dir / f"{model_name}.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save metadata
    metadata = {
        'version': version,
        'model_type': 'xgboost_regressor',
        'history_window_size': config.history_window_size,
        'prediction_horizon': config.prediction_horizon,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'metrics': metrics,
        'xgboost_params': model.get_params(),
    }

    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {metadata_path}")

    # Create symlink to latest model
    latest_model = output_dir / "latest.json"
    latest_metadata = output_dir / "latest_metadata.json"

    if latest_model.exists():
        latest_model.unlink()
    if latest_metadata.exists():
        latest_metadata.unlink()

    latest_model.symlink_to(model_path.name)
    latest_metadata.symlink_to(metadata_path.name)

    return model_path


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train XGBoost cognitive load forecaster")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (uses defaults if not provided)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("XGBoost Cognitive Load Forecaster Training")
    print("=" * 60)

    # Load training config from file or use defaults
    if args.config:
        print(f"Loading config from: {args.config}")
        system_config = SystemConfig.from_file(args.config)
        config = system_config.training
    else:
        print("Using default TrainingConfig")
        config = TrainingConfig()

    # Prepare dataset
    dataset = prepare_dataset(config=config)

    # Train model
    model = train_model(
        dataset['X_train'],
        dataset['y_train'],
        dataset['X_val'],
        dataset['y_val'],
        dataset['feature_names'],
        config=config,
    )

    # Evaluate
    metrics = evaluate_model(
        model,
        dataset['X_test'],
        dataset['y_test'],
    )

    # Feature importance
    importance = get_feature_importance(model, dataset['feature_names'])

    # Save model
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "models" / "trained"

    save_model(model, metrics, dataset['feature_names'], output_dir, config=config)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
