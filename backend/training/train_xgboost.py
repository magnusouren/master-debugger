"""
Train XGBoost model for WindowFeatures forecasting.

Usage:
    python -m backend.training.train_xgboost
    python -m backend.training.train_xgboost --config backend/config.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    raise SystemExit(1)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.types import SystemConfig, TrainingConfig
from backend.training.dataset import prepare_dataset


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    config: TrainingConfig,
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor with early stopping.
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
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)

    if best_iteration is not None:
        print(f"\nBest iteration: {best_iteration}")
    if best_score is not None:
        print(f"Best validation score: {best_score:.4f}")

    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_columns: List[str],
) -> Dict[str, Any]:
    """
    Evaluate model on test set.

    Returns overall metrics plus per-target metrics.
    Also logs normalized error metrics so RMSE can be interpreted
    relative to the target distribution.
    """
    print("\n--- Evaluation on Test Set ---")

    y_pred = model.predict(X_test)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Overall regression metrics across all outputs
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred, multioutput="uniform_average"))

    # Global descriptive stats over all targets combined
    overall_std = float(np.std(y_test))
    overall_min = float(np.min(y_test))
    overall_max = float(np.max(y_test))
    overall_range = float(overall_max - overall_min)

    overall_nrmse_by_std = float(rmse / overall_std) if overall_std > 0 else None
    overall_nrmse_by_range = float(rmse / overall_range) if overall_range > 0 else None

    print(f"Overall RMSE:          {rmse:.4f}")
    print(f"Overall MAE:           {mae:.4f}")
    print(f"Overall R²:            {r2:.4f}")
    print(f"Overall target std:    {overall_std:.4f}")
    print(f"Overall target range:  {overall_range:.4f}")
    if overall_nrmse_by_std is not None:
        print(f"Overall RMSE / std:    {overall_nrmse_by_std:.4f}")
    if overall_nrmse_by_range is not None:
        print(f"Overall RMSE / range:  {overall_nrmse_by_range:.4f}")

    # Per-target metrics
    per_target: Dict[str, Dict[str, float]] = {}
    print("\nPer-target metrics:")
    for idx, col in enumerate(target_columns):
        yt = y_test[:, idx]
        yp = y_pred[:, idx]

        col_mse = mean_squared_error(yt, yp)
        col_rmse = float(np.sqrt(col_mse))
        col_mae = float(mean_absolute_error(yt, yp))
        col_r2 = float(r2_score(yt, yp))

        col_mean = float(np.mean(yt))
        col_std = float(np.std(yt))
        col_min = float(np.min(yt))
        col_max = float(np.max(yt))
        col_range = float(col_max - col_min)

        col_nrmse_by_std = float(col_rmse / col_std) if col_std > 0 else None
        col_nrmse_by_range = float(col_rmse / col_range) if col_range > 0 else None

        per_target[col] = {
            "rmse": col_rmse,
            "mae": col_mae,
            "r2": col_r2,
            "mean": col_mean,
            "std": col_std,
            "min": col_min,
            "max": col_max,
            "range": col_range,
            "nrmse_by_std": col_nrmse_by_std,
            "nrmse_by_range": col_nrmse_by_range,
        }

        print(
            f"  {col:<24} "
            f"RMSE={col_rmse:>8.4f} "
            f"MAE={col_mae:>8.4f} "
            f"R²={col_r2:>8.4f} "
            f"std={col_std:>8.4f} "
            f"RMSE/std={col_nrmse_by_std:>8.4f}"
            if col_nrmse_by_std is not None
            else
            f"  {col:<24} "
            f"RMSE={col_rmse:>8.4f} "
            f"MAE={col_mae:>8.4f} "
            f"R²={col_r2:>8.4f} "
            f"std={col_std:>8.4f} "
            f"RMSE/std={'None':>8}"
        )

    return {
        "overall": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "std": overall_std,
            "min": overall_min,
            "max": overall_max,
            "range": overall_range,
            "nrmse_by_std": overall_nrmse_by_std,
            "nrmse_by_range": overall_nrmse_by_range,
        },
        "per_target": per_target,
    }


def get_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: List[str],
    top_n: int = 20,
) -> Dict[str, float]:
    """
    Get top feature importances.
    """
    importances = np.asarray(model.feature_importances_)

    # Multi-output models may return one vector per target
    if importances.ndim == 2:
        importances = importances.mean(axis=0)

    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n--- Top {top_n} Feature Importances ---")
    importance_dict: Dict[str, float] = {}

    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")
        importance_dict[feature_names[i]] = float(importances[i])

    return importance_dict


def save_model(
    model: xgb.XGBRegressor,
    metrics: Dict[str, Any],
    feature_names: List[str],
    target_columns: List[str],
    input_columns: List[str],
    normalization_mode: Optional[str],
    participant_normalizers: Optional[Dict[str, Any]],
    output_dir: Path,
    config: TrainingConfig,
) -> Path:
    """
    Save trained model and metadata.

    Returns path to saved model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgb_forecaster_{version}"

    model_path = output_dir / f"{model_name}.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    metadata = {
        "version": version,
        "model_type": "xgboost_regressor",
        "task": "sequence_to_one_windowfeatures_forecasting",
        "history_window_size": config.history_window_size,
        "prediction_horizon": config.prediction_horizon,
        "input_columns": input_columns,
        "target_columns": target_columns,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "metrics": metrics,
        "xgboost_params": model.get_params(),
        "target_type": "delta",
        "normalization_mode": normalization_mode,
    }

    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {metadata_path}")

    latest_model = output_dir / "latest.json"
    latest_metadata = output_dir / "latest_metadata.json"

    if latest_model.is_symlink() or latest_model.exists():
        latest_model.unlink()
    if latest_metadata.is_symlink() or latest_metadata.exists():
        latest_metadata.unlink()

    shutil.copy2(model_path, latest_model)
    shutil.copy2(metadata_path, latest_metadata)

    return model_path


def main() -> None:
    """
    Main training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Train XGBoost WindowFeatures forecaster"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="backend/config.yaml",
        help="Path to config.yaml file (uses defaults if not provided)",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Directory to load/save participant split files (train/val/test).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("XGBoost WindowFeatures Forecaster Training")
    print("=" * 60)

    if args.config:
        print(f"Loading config from: {args.config}")
        system_config = SystemConfig.from_file(args.config)
        config = system_config.training
    else:
        print("Using default TrainingConfig")
        config = TrainingConfig()

    split_dir = Path(args.split_dir) if args.split_dir else None

    dataset = prepare_dataset(config=config, split_dir=split_dir)

    model = train_model(
        dataset["X_train"],
        dataset["y_train"],
        dataset["X_val"],
        dataset["y_val"],
        dataset["feature_names"],
        config=config,
    )

    metrics = evaluate_model(
        model,
        dataset["X_test"],
        dataset["y_test"],
        dataset["target_columns"],
    )

    get_feature_importance(model, dataset["feature_names"])

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "models" / "trained"

    save_model(
        model=model,
        metrics=metrics,
        feature_names=dataset["feature_names"],
        target_columns=dataset["target_columns"],
        input_columns=dataset["input_columns"],
        normalization_mode=dataset.get("normalization_mode"),
        participant_normalizers=dataset.get("participant_normalizers_serialized"),
        output_dir=output_dir,
        config=config,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
