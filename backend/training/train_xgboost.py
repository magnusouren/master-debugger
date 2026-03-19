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
import shutil
from typing import Dict
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    exit(1)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.types import TrainingConfig, SystemConfig
from backend.training.dataset import prepare_dataset
from backend.models.forecast_feature_schema import TARGET_COLUMNS, compute_score_from_target_components


def fit_target_scaler(y_train: np.ndarray) -> dict:
    """
    Fit per-target standardization stats on training targets only.
    """
    mean = np.mean(y_train, axis=0)
    std = np.std(y_train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def transform_targets(y: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply per-target standardization."""
    return (y - scaler["mean"]) / scaler["std"]


def inverse_transform_targets(y_scaled: np.ndarray, scaler: dict) -> np.ndarray:
    """Inverse per-target standardization."""
    return y_scaled * scaler["std"] + scaler["mean"]


def _compute_score_metrics(y_true_score: np.ndarray, y_pred_score: np.ndarray) -> dict:
    """Compute regression metrics on score space."""
    mse = mean_squared_error(y_true_score, y_pred_score)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_score, y_pred_score)
    r2 = r2_score(y_true_score, y_pred_score)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def _print_naive_baselines(y_test_score: np.ndarray, X_test: np.ndarray) -> dict:
    """
    Print naive score-space baselines for calibration.

    Baseline A: global mean score
    Baseline B: persistence from latest contributor window (t-0 average of 5 contribs)
    """
    print("\n--- Naive Baselines (score space) ---")

    global_mean_pred = np.full_like(y_test_score, y_test_score.mean())
    global_mean_metrics = _compute_score_metrics(y_test_score, global_mean_pred)
    print(
        "Global mean baseline -> "
        f"RMSE: {global_mean_metrics['rmse']:.4f}, "
        f"MAE: {global_mean_metrics['mae']:.4f}, "
        f"R²: {global_mean_metrics['r2']:.4f}"
    )

    # Flattened feature vector stores the latest 5 contributor inputs at the tail.
    t0_persist_pred = X_test[:, -5:].mean(axis=1)
    persistence_metrics = _compute_score_metrics(y_test_score, t0_persist_pred)
    print(
        "Persistence (t-0 contrib mean) -> "
        f"RMSE: {persistence_metrics['rmse']:.4f}, "
        f"MAE: {persistence_metrics['mae']:.4f}, "
        f"R²: {persistence_metrics['r2']:.4f}"
    )

    return {
        "global_mean": global_mean_metrics,
        "persistence_t0": persistence_metrics,
    }


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    config: TrainingConfig,
) -> Dict[str, xgb.XGBRegressor]:
    """
    Train one XGBoost regressor per target with early stopping.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names
        config: TrainingConfig with hyperparameters

    Returns:
        Dictionary of trained models keyed by target column name
    """
    print("\n--- Training XGBoost ---")
    models: Dict[str, xgb.XGBRegressor] = {}

    for target_idx, target_name in enumerate(TARGET_COLUMNS):
        print(f"\nTarget: {target_name}")

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
            y_train[:, target_idx],
            eval_set=[(X_val, y_val[:, target_idx])],
            verbose=False,
        )

        print(f"Best iteration: {model.best_iteration}")
        print(f"Best validation score: {model.best_score:.4f}")
        models[target_name] = model

    return models


def evaluate_model(
    models: Dict[str, xgb.XGBRegressor],
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: dict,
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

    y_pred_scaled = np.column_stack(
        [models[target].predict(X_test) for target in TARGET_COLUMNS]
    )
    y_pred = inverse_transform_targets(y_pred_scaled, target_scaler)

    # Convert component vectors -> strict 5x0.2 score for primary metrics.
    y_test_score = np.array([
        compute_score_from_target_components({
            key: float(y_test[i, idx]) for idx, key in enumerate(TARGET_COLUMNS)
        })
        for i in range(len(y_test))
    ])
    y_pred_score = np.array([
        compute_score_from_target_components({
            key: float(y_pred[i, idx]) for idx, key in enumerate(TARGET_COLUMNS)
        })
        for i in range(len(y_pred))
    ])

    # Regression metrics
    score_metrics = _compute_score_metrics(y_test_score, y_pred_score)
    rmse = score_metrics["rmse"]
    mae = score_metrics["mae"]
    r2 = score_metrics["r2"]

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # Per-target regression metrics on raw contributor values.
    print("\n--- Per-Target Regression Metrics ---")
    per_target_metrics = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        target_metrics = _compute_score_metrics(
            y_test[:, idx],
            y_pred[:, idx],
        )
        per_target_metrics[target] = target_metrics
        print(
            f"{target}: "
            f"RMSE={target_metrics['rmse']:.4f}, "
            f"MAE={target_metrics['mae']:.4f}, "
            f"R²={target_metrics['r2']:.4f}"
        )

    # Binary classification metrics (above/below threshold)
    y_test_binary = (y_test_score >= threshold).astype(int)
    y_pred_binary = (y_pred_score >= threshold).astype(int)

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

    baseline_metrics = _print_naive_baselines(y_test_score, X_test)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_target': per_target_metrics,
        'naive_baselines': baseline_metrics,
    }


def get_feature_importance(
    models: Dict[str, xgb.XGBRegressor],
    feature_names: list,
    top_n: int = 20
) -> dict:
    """Get top averaged feature importances across target models."""
    all_importances = [
        np.asarray(models[target].feature_importances_)
        for target in TARGET_COLUMNS
        if target in models
    ]
    importances = np.mean(np.vstack(all_importances), axis=0)
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n--- Top {top_n} Feature Importances ---")
    importance_dict = {}
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")
        importance_dict[feature_names[i]] = float(importances[i])

    return importance_dict


def save_model(
    models: Dict[str, xgb.XGBRegressor],
    metrics: dict,
    feature_names: list,
    output_dir: Path,
    config: TrainingConfig,
    target_scaler: dict,
) -> Path:
    """
    Save trained model and metadata.

    Returns path to saved model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate version string
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgb_forecaster_{version}"

    # Save one model file per target.
    model_files: dict[str, str] = {}
    for target in TARGET_COLUMNS:
        model = models[target]
        file_name = f"{model_name}_{target}.json"
        model_path = output_dir / file_name
        model.save_model(str(model_path))
        model_files[target] = file_name
        print(f"Model ({target}) saved to: {model_path}")

    # Save metadata
    metadata = {
        'version': version,
        'model_type': 'xgboost_regressor_ensemble',
        'target_columns': TARGET_COLUMNS,
        'model_files': model_files,
        'history_window_size': config.history_window_size,
        'prediction_horizon': config.prediction_horizon,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'metrics': metrics,
        'xgboost_params': models[TARGET_COLUMNS[0]].get_params(),
        'target_scaler': {
            'columns': TARGET_COLUMNS,
            'mean': [float(v) for v in target_scaler['mean']],
            'std': [float(v) for v in target_scaler['std']],
        },
    }

    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {metadata_path}")

    # Write latest model/metadata as real files (not symlinks) for
    # cross-platform compatibility, especially on Windows.
    latest_model = output_dir / "latest.json"
    latest_metadata = output_dir / "latest_metadata.json"
    primary_model_path = output_dir / model_files[TARGET_COLUMNS[0]]

    # Remove existing files/symlinks first
    if latest_model.is_symlink() or latest_model.exists():
        latest_model.unlink()
    if latest_metadata.is_symlink() or latest_metadata.exists():
        latest_metadata.unlink()

    shutil.copy2(primary_model_path, latest_model)
    shutil.copy2(metadata_path, latest_metadata)

    return primary_model_path


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train XGBoost cognitive load forecaster")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (uses defaults if not provided)"
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Directory to load/save participant split files (train/val/test).",
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
    split_dir = Path(args.split_dir) if args.split_dir else None
    dataset = prepare_dataset(config=config, split_dir=split_dir)

    # Train model
    target_scaler = fit_target_scaler(dataset['y_train'])
    y_train_scaled = transform_targets(dataset['y_train'], target_scaler)
    y_val_scaled = transform_targets(dataset['y_val'], target_scaler)

    models = train_models(
        dataset['X_train'],
        y_train_scaled,
        dataset['X_val'],
        y_val_scaled,
        dataset['feature_names'],
        config=config,
    )

    # Evaluate
    metrics = evaluate_model(
        models,
        dataset['X_test'],
        dataset['y_test'],
        target_scaler=target_scaler,
    )

    # Feature importance
    importance = get_feature_importance(models, dataset['feature_names'])

    # Save model
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "models" / "trained"

    save_model(
        models,
        metrics,
        dataset['feature_names'],
        output_dir,
        config=config,
        target_scaler=target_scaler,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
