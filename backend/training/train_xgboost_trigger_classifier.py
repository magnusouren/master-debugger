"""
Train XGBoost binary classifier for future reactive feedback triggering.

This script keeps the regression pipeline untouched and builds a parallel
classification pipeline where the target is:

    y = 1 if the FUTURE smoothed reactive score at time t + horizon_seconds
            is outside participant-specific trigger bounds
            for >= N consecutive observed windows
        0 otherwise

Key design goals:
- Reuse existing ReactiveTool rule-based scoring, smoothing, baseline-aware
  normalization, and trigger bound calibration logic as-is.
- Predict an upcoming trigger event some number of seconds ahead.
- Avoid leakage from baseline-state into task-state after calibration.

Usage:
    python -m backend.training.train_xgboost_future_trigger_classifier
    python -m backend.training.train_xgboost_future_trigger_classifier --config backend/config.yaml
    python -m backend.training.train_xgboost_future_trigger_classifier --prediction-horizon-seconds 10
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    raise SystemExit(1)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.layers.reactive_tool import ReactiveTool
from backend.services.logger_service import LoggerService
from backend.training.dataset import (
    DEFAULT_INPUT_COLUMNS,
    load_processed_data,
    row_to_window_features,
    split_by_participant,
)
from backend.types import ReactiveToolConfig, SystemConfig, TrainingConfig, WindowFeatures


DEFAULT_CONSECUTIVE_OUTSIDE_WINDOWS = 2
DEFAULT_BASELINE_DURATION_SECONDS = 60.0
DEFAULT_PREDICTION_HORIZON_SECONDS = 10.0
MIN_WINDOWS_FOR_REACTIVE_SCORING = 3


@dataclass
class ParticipantBuildResult:
    participant_id: str
    n_observed_windows: int
    n_baseline_windows: int
    n_scored_windows: int
    n_samples: int
    n_positive: int
    positive_ratio: float
    window_step_seconds: float
    horizon_windows: int
    trigger_bounds: Dict[str, float | int | str]
    trigger_lower: Optional[float]
    trigger_upper: Optional[float]
    trigger_width: Optional[float]


@dataclass
class LabelBuildOutputs:
    X: np.ndarray
    y: np.ndarray
    participant_ids: List[str]
    feature_names: List[str]
    input_columns: List[str]
    participant_summaries: List[ParticipantBuildResult]
    skipped_participants: List[Tuple[str, str]]


def _to_float(value: Any) -> Optional[float]:
    """Safely convert scalar to finite float."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(f):
        return None
    return f


def _get_input_columns(config: Optional[TrainingConfig]) -> List[str]:
    """Resolve classifier input columns with fallback defaults."""
    if config is not None:
        cols = getattr(config, "input_feature_columns", None)
        if cols:
            return list(cols)
    return list(DEFAULT_INPUT_COLUMNS)


def _validate_columns_exist(df: pd.DataFrame, columns: List[str], label: str) -> None:
    """Raise clear error if expected columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {label} columns in processed dataframe: {missing}\n"
            f"Available columns are: {list(df.columns)}"
        )


def _filter_observed_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep observed windows only, ignoring predicted windows when metadata exists."""
    if "is_predicted" in df.columns:
        mask = ~df["is_predicted"].fillna(False).astype(bool)
        return df.loc[mask].copy()

    if "source_type" in df.columns:
        mask = df["source_type"].fillna("observed_features") != "predicted_features"
        return df.loc[mask].copy()

    return df.copy()


def _compute_feature_names(history_size: int, input_columns: List[str]) -> List[str]:
    """Build flattened feature names aligned with history-window flattening."""
    names: List[str] = []
    for t in range(history_size):
        for col in input_columns:
            names.append(f"t-{history_size - t - 1}_{col}")
    return names


def _get_window_step_seconds(group: pd.DataFrame) -> float:
    """Estimate observed window step in seconds for a participant group."""
    if len(group) < 2:
        return 0.5

    deltas = group["window_end"].diff().dropna().to_numpy(dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    deltas = deltas[deltas > 0]

    if len(deltas) == 0:
        return 0.5

    return float(np.median(deltas))


def _baseline_window_count(group: pd.DataFrame, baseline_duration_seconds: float) -> int:
    """Convert baseline duration to number of windows for a participant group."""
    step = _get_window_step_seconds(group)
    if step <= 0:
        step = 0.5
    return max(int(round(baseline_duration_seconds / step)), MIN_WINDOWS_FOR_REACTIVE_SCORING)


def _prediction_horizon_windows(group: pd.DataFrame, prediction_horizon_seconds: float) -> int:
    """Convert future prediction horizon in seconds to approximate number of windows."""
    step = _get_window_step_seconds(group)
    if step <= 0:
        step = 0.5
    return max(int(round(prediction_horizon_seconds / step)), 1)


def _row_to_observed_window(
    row: pd.Series,
    enabled_metrics: List[str],
    participant_id: str,
    row_pos: int,
) -> WindowFeatures:
    """Convert processed row into WindowFeatures while preserving observed-source semantics."""
    wf = row_to_window_features(row, enabled_metrics=enabled_metrics)
    wf.window_id = f"{participant_id}_obs_{row_pos}"
    wf.is_predicted = bool(row.get("is_predicted", False))
    return wf


def _build_outside_bounds_labels(outside_flags: List[bool], consecutive_n: int) -> List[int]:
    """
    Convert outside-bounds booleans into consecutive-window trigger labels.

    Example for consecutive_n=2:
    outside = [F, T, T, T, F, T, T]
    labels  = [0, 0, 1, 1, 0, 0, 1]
    """
    labels: List[int] = []
    run_len = 0

    for outside in outside_flags:
        if outside:
            run_len += 1
        else:
            run_len = 0

        labels.append(1 if run_len >= consecutive_n else 0)

    return labels


def _make_task_reactive_tool(
    baseline,
    trigger_bounds: Dict[str, float | int | str],
    reactive_config: ReactiveToolConfig,
    logger: LoggerService,
) -> ReactiveTool:
    """
    Create a fresh ReactiveTool for task replay after calibration.

    This avoids leaking baseline windows and baseline score-history into the
    task replay phase.
    """
    task_tool = ReactiveTool(config=reactive_config, logger=logger)
    task_tool.set_baseline(baseline)

    # Persist calibrated trigger bounds into the new instance.
    task_tool._feedback_trigger_bounds = dict(trigger_bounds)  # intentional internal reuse
    task_tool.start()
    return task_tool


def _reconstruct_future_trigger_labels_for_participant(
    participant_id: str,
    group: pd.DataFrame,
    reactive_config: ReactiveToolConfig,
    baseline_duration_seconds: float,
    consecutive_outside_windows: int,
) -> Tuple[
    Optional[Dict[int, int]],
    Optional[Dict[int, float]],
    Optional[Dict[str, float | int | str]],
    str,
    int,
    float,
]:
    """
    Reconstruct reactive task scores and future trigger labels for one participant.

    Returns:
        label_by_row_pos,
        score_by_row_pos,
        trigger_bounds,
        skip_reason,
        baseline_count,
        window_step_seconds
    """
    if len(group) < MIN_WINDOWS_FOR_REACTIVE_SCORING:
        return None, None, None, "too_few_observed_windows", 0, 0.0

    logger = LoggerService(experiment_level="ERROR", system_level="ERROR")
    baseline_tool = ReactiveTool(config=reactive_config, logger=logger)
    baseline_tool.start()

    baseline_count = _baseline_window_count(group, baseline_duration_seconds)
    window_step_seconds = _get_window_step_seconds(group)

    if len(group) <= baseline_count:
        return None, None, None, "insufficient_windows_after_baseline", baseline_count, window_step_seconds

    enabled_metrics = list(group.iloc[-1].get("enabled_metrics", []))
    if not enabled_metrics:
        enabled_metrics = [
            "fixation_duration",
            "saccade_amplitude",
            "pupil_diameter",
            "blink_rate",
            "data_quality",
            "ipi",
        ]

    # Step 1: baseline recording on earliest observed windows.
    baseline_rows = group.iloc[:baseline_count]
    baseline_tool.start_baseline_recording(participant_id)

    for pos, (_, row) in enumerate(baseline_rows.iterrows()):
        wf = _row_to_observed_window(row, enabled_metrics, participant_id, pos)
        baseline_tool.add_features(wf)

    baseline = baseline_tool.stop_baseline_recording(participant_id)
    if baseline is None:
        return None, None, None, "missing_or_invalid_baseline", baseline_count, window_step_seconds

    # Step 2: activate baseline, then calibrate trigger bounds in calibrated score space.
    baseline_tool.set_baseline(baseline)
    trigger_bounds = baseline_tool.calibrate_feedback_trigger_bounds_from_baseline_windows()
    if trigger_bounds is None:
        return None, None, None, "missing_trigger_bounds", baseline_count, window_step_seconds

    lower = _to_float(trigger_bounds.get("lower"))
    upper = _to_float(trigger_bounds.get("upper"))
    if lower is None or upper is None:
        return None, None, None, "invalid_trigger_bounds", baseline_count, window_step_seconds

    # Step 3: use a fresh tool instance for task replay so baseline state does not leak.
    task_tool = _make_task_reactive_tool(
        baseline=baseline,
        trigger_bounds=trigger_bounds,
        reactive_config=reactive_config,
        logger=logger,
    )

    post_rows = group.iloc[baseline_count:]
    scored_row_positions: List[int] = []
    scores: List[float] = []
    outside_flags: List[bool] = []

    for pos, (_, row) in enumerate(post_rows.iterrows(), start=baseline_count):
        wf = _row_to_observed_window(row, enabled_metrics, participant_id, pos)
        task_tool.add_features(wf)

        estimate = task_tool.get_latest_estimate()
        if estimate is None:
            continue

        score = _to_float(estimate.score.score)
        if score is None:
            continue

        scored_row_positions.append(pos)
        scores.append(score)
        outside_flags.append(score < lower or score > upper)

    if len(scores) < MIN_WINDOWS_FOR_REACTIVE_SCORING:
        return None, None, None, "insufficient_scored_windows", baseline_count, window_step_seconds

    labels = _build_outside_bounds_labels(outside_flags, consecutive_outside_windows)

    score_by_row_pos = {
        row_pos: score for row_pos, score in zip(scored_row_positions, scores)
    }
    label_by_row_pos = {
        row_pos: label for row_pos, label in zip(scored_row_positions, labels)
    }

    return (
        label_by_row_pos,
        score_by_row_pos,
        trigger_bounds,
        "",
        baseline_count,
        window_step_seconds,
    )


def _build_future_history_samples_for_participant(
    participant_id: str,
    group: pd.DataFrame,
    input_columns: List[str],
    history_size: int,
    horizon_windows: int,
    label_by_row_pos: Dict[int, int],
    baseline_count: int,
    max_missing_fraction_per_block: float = 0.10,
    debug_print: bool = True,
) -> Tuple[List[List[float]], List[int]]:
    """
    Build flattened history-window samples for future trigger prediction.

    X uses observed history ending at current time t.
    y is the trigger label at future time t + horizon_windows.

    Improvements over the stricter version:
    - Logs exactly why candidate samples are skipped
    - Allows limited missingness inside a history block
    - Imputes missing values using:
        1. forward fill inside the block
        2. backward fill inside the block
        3. participant-level column median fallback
    - Still skips blocks that are too sparse

    Args:
        participant_id: Participant identifier (for logging)
        group: Participant dataframe sorted by time
        input_columns: Feature columns used as input
        history_size: Number of past windows in X
        horizon_windows: Future offset in windows for y
        label_by_row_pos: Mapping from row position -> future label
        baseline_count: Number of baseline windows at start
        max_missing_fraction_per_block: Max allowed fraction of missing values in one block
        debug_print: Whether to print per-participant diagnostics

    Returns:
        X_local, y_local
    """
    X_local: List[List[float]] = []
    y_local: List[int] = []

    if not label_by_row_pos:
        if debug_print:
            print(f"[SAMPLE_BUILD] pid={participant_id} no labels available")
        return X_local, y_local

    # Keep only needed columns and coerce to numeric
    matrix_df = group[input_columns].copy()
    for col in input_columns:
        matrix_df[col] = pd.to_numeric(matrix_df[col], errors="coerce")

    # Participant-level fallback medians
    participant_medians = matrix_df.median(axis=0, skipna=True)

    # If any median is still NaN (column completely missing for this participant),
    # fall back to global safe value 0.0 for that column
    participant_medians = participant_medians.fillna(0.0)

    matrix = matrix_df.to_numpy(dtype=np.float32, copy=True)

    # Diagnostics
    candidate_positions = 0
    skipped_missing_future_label = 0
    skipped_short_history = 0
    skipped_too_many_missing = 0
    accepted_samples = 0

    start_idx = max(history_size - 1, baseline_count)
    max_current_idx = len(group) - 1 - horizon_windows

    for current_idx in range(start_idx, max_current_idx + 1):
        candidate_positions += 1
        target_idx = current_idx + horizon_windows

        # Need a valid future label
        if target_idx not in label_by_row_pos:
            skipped_missing_future_label += 1
            continue

        hist_start = current_idx - history_size + 1
        hist_end = current_idx + 1  # exclusive upper bound in slicing

        if hist_start < 0:
            skipped_short_history += 1
            continue

        history_block = matrix[hist_start:hist_end, :].copy()

        if history_block.shape[0] != history_size:
            skipped_short_history += 1
            continue

        # Missingness check before imputation
        missing_mask = ~np.isfinite(history_block)
        missing_fraction = float(np.mean(missing_mask))

        if missing_fraction > max_missing_fraction_per_block:
            skipped_too_many_missing += 1
            continue

        # Impute within block:
        # 1) forward fill
        # 2) backward fill
        # 3) participant median fallback
        block_df = pd.DataFrame(history_block, columns=input_columns)

        block_df = block_df.ffill().bfill()

        for col in input_columns:
            if block_df[col].isna().any():
                block_df[col] = block_df[col].fillna(participant_medians[col])

        # Final safety check
        if block_df.isna().any().any():
            skipped_too_many_missing += 1
            continue

        flattened = block_df.to_numpy(dtype=np.float32).reshape(-1).tolist()
        X_local.append(flattened)
        y_local.append(int(label_by_row_pos[target_idx]))
        accepted_samples += 1

    if debug_print:
        pos_ratio = (sum(y_local) / len(y_local)) if y_local else 0.0
        print(
            f"[SAMPLE_BUILD] pid={participant_id} "
            f"candidates={candidate_positions} "
            f"accepted={accepted_samples} "
            f"skip_future={skipped_missing_future_label} "
            f"skip_short_history={skipped_short_history} "
            f"skip_too_many_missing={skipped_too_many_missing} "
            f"pos_ratio={pos_ratio:.4f}"
        )

    return X_local, y_local


def build_future_trigger_classification_dataset(
    df: pd.DataFrame,
    input_columns: List[str],
    history_size: int,
    prediction_horizon_seconds: float,
    reactive_config: ReactiveToolConfig,
    baseline_duration_seconds: float,
    consecutive_outside_windows: int,
) -> LabelBuildOutputs:
    """Build complete classifier dataset for future trigger prediction."""
    X_all: List[List[float]] = []
    y_all: List[int] = []
    participant_ids: List[str] = []
    participant_summaries: List[ParticipantBuildResult] = []
    skipped_participants: List[Tuple[str, str]] = []

    grouped = df.groupby("participant_id", sort=True)

    for participant_id, participant_df in grouped:
        group = participant_df.sort_values(["trial", "window_start"]).reset_index(drop=True)

        (
            label_by_row_pos,
            score_by_row_pos,
            trigger_bounds,
            skip_reason,
            baseline_count,
            window_step_seconds,
        ) = _reconstruct_future_trigger_labels_for_participant(
            participant_id=str(participant_id),
            group=group,
            reactive_config=reactive_config,
            baseline_duration_seconds=baseline_duration_seconds,
            consecutive_outside_windows=consecutive_outside_windows,
        )

        if (
            label_by_row_pos is None
            or score_by_row_pos is None
            or trigger_bounds is None
        ):
            skipped_participants.append((str(participant_id), skip_reason))
            print(f"[WARN] Skipping participant {participant_id}: {skip_reason}")
            continue

        horizon_windows = _prediction_horizon_windows(group, prediction_horizon_seconds)

        X_local, y_local = _build_future_history_samples_for_participant(
            participant_id=str(participant_id),
            group=group,
            input_columns=input_columns,
            history_size=history_size,
            horizon_windows=horizon_windows,
            label_by_row_pos=label_by_row_pos,
            baseline_count=baseline_count,
            max_missing_fraction_per_block=0.10,
            debug_print=True,
        )

        if not X_local:
            skipped_participants.append((str(participant_id), "no_valid_history_samples"))
            print(f"[WARN] Skipping participant {participant_id}: no_valid_history_samples")
            continue

        X_all.extend(X_local)
        y_all.extend(y_local)
        participant_ids.extend([str(participant_id)] * len(X_local))

        trigger_lower = _to_float(trigger_bounds.get("lower"))
        trigger_upper = _to_float(trigger_bounds.get("upper"))
        trigger_width = None
        if trigger_lower is not None and trigger_upper is not None:
            trigger_width = float(trigger_upper - trigger_lower)

        if trigger_width is None or trigger_width <= 0.05:
            skipped_participants.append((str(participant_id), "trigger_width_too_small"))
            print(
                f"[WARN] Skipping participant {participant_id}: "
                f"trigger_width_too_small ({trigger_width})"
            )
            continue

        n_positive = int(sum(y_local))
        positive_ratio = float(n_positive / len(X_local)) if X_local else 0.0

        participant_summaries.append(
            ParticipantBuildResult(
                participant_id=str(participant_id),
                n_observed_windows=int(len(group)),
                n_baseline_windows=int(baseline_count),
                n_scored_windows=int(len(score_by_row_pos)),
                n_samples=int(len(X_local)),
                n_positive=n_positive,
                positive_ratio=positive_ratio,
                window_step_seconds=float(window_step_seconds),
                horizon_windows=int(horizon_windows),
                trigger_bounds=trigger_bounds,
                trigger_lower=trigger_lower,
                trigger_upper=trigger_upper,
                trigger_width=trigger_width,
            )
        )

    feature_names = _compute_feature_names(history_size=history_size, input_columns=input_columns)

    return LabelBuildOutputs(
        X=np.asarray(X_all, dtype=np.float32),
        y=np.asarray(y_all, dtype=np.int32),
        participant_ids=participant_ids,
        feature_names=feature_names,
        input_columns=input_columns,
        participant_summaries=participant_summaries,
        skipped_participants=skipped_participants,
    )


def _class_balance(y: np.ndarray) -> Dict[str, float]:
    """Return class counts and ratios."""
    if len(y) == 0:
        return {
            "n_samples": 0,
            "n_negative": 0,
            "n_positive": 0,
            "positive_ratio": 0.0,
        }

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return {
        "n_samples": int(len(y)),
        "n_negative": n_neg,
        "n_positive": n_pos,
        "positive_ratio": float(n_pos / max(1, len(y))),
    }


def _compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute XGBoost scale_pos_weight for class imbalance handling."""
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def _select_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Select probability threshold on validation data using best F1.

    Returns 0.5 if the curve is degenerate.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5

    # precision/recall have length len(thresholds)+1
    f1_scores = []
    for p, r in zip(precision[:-1], recall[:-1]):
        denom = p + r
        f1_scores.append(0.0 if denom <= 0 else (2.0 * p * r / denom))

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    scale_pos_weight: float,
) -> xgb.XGBClassifier:
    """Train XGBoost binary classifier with early stopping."""
    print("\n--- Training XGBoost Future Trigger Classifier ---")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
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
        scale_pos_weight=scale_pos_weight,
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
        print(f"Best iteration: {best_iteration}")
    if best_score is not None:
        print(f"Best validation score (AUCPR): {best_score:.6f}")

    return model


def evaluate_classifier(
    model: xgb.XGBClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate classifier performance.

    Threshold is selected on validation data and then applied to test data.
    """
    print("\n--- Evaluating Future Trigger Classifier ---")

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold = _select_best_threshold_by_f1(y_val, val_prob)
    print(f"Selected threshold from validation set: {best_threshold:.4f}")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(np.int32)

    metrics: Dict[str, Any] = {
        "selected_threshold": float(best_threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    unique_labels = np.unique(y_test)
    if len(unique_labels) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        print("[WARN] ROC-AUC / PR-AUC not available (only one class present in test labels)")

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    if metrics["pr_auc"] is not None:
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"Confusion matrix [TN, FP; FN, TP]: {metrics['confusion_matrix']}")

    return metrics


def get_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    top_n: int = 20,
) -> Dict[str, float]:
    """Print and return top feature importances."""
    importances = np.asarray(model.feature_importances_)
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n--- Top {top_n} Feature Importances ---")
    importance_dict: Dict[str, float] = {}

    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")
        importance_dict[feature_names[i]] = float(importances[i])

    return importance_dict


def save_model_and_metadata(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    input_columns: List[str],
    output_dir: Path,
    history_window_size: int,
    prediction_horizon_seconds: float,
    consecutive_outside_windows: int,
    baseline_duration_seconds: float,
    reactive_config: ReactiveToolConfig,
    metrics: Dict[str, Any],
    class_balance: Dict[str, Dict[str, float]],
    participant_summaries: List[ParticipantBuildResult],
    skipped_participants: List[Tuple[str, str]],
) -> Path:
    """Save trained classifier model and metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgb_future_trigger_classifier_{version}"

    model_path = output_dir / f"{model_name}.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    metadata = {
        "version": version,
        "model_type": "xgboost_classifier",
        "task": "future_reactive_feedback_trigger_binary_classification",
        "history_window_size": history_window_size,
        "prediction_horizon_seconds": prediction_horizon_seconds,
        "input_columns": input_columns,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "label_rule": {
            "description": "future_smoothed_score_outside_calibrated_bounds_for_n_consecutive_observed_windows",
            "consecutive_outside_windows": consecutive_outside_windows,
            "minimum_windows_for_scoring": MIN_WINDOWS_FOR_REACTIVE_SCORING,
            "observed_windows_only": True,
            "ignore_cooldown_and_delivery_logic": True,
            "predict_future_trigger": True,
        },
        "calibration": {
            "baseline_duration_seconds": baseline_duration_seconds,
            "baseline_bounds_source": "ReactiveTool.calibrate_feedback_trigger_bounds_from_baseline_windows",
            "baseline_aware_normalization": True,
            "reactive_window_size_seconds": reactive_config.window_size_seconds,
            "score_smoothing_factor": reactive_config.score_smoothing_factor,
            "model_type": reactive_config.model_type,
        },
        "class_balance": class_balance,
        "metrics": metrics,
        "participant_summaries": [
            {
                "participant_id": p.participant_id,
                "n_observed_windows": p.n_observed_windows,
                "n_baseline_windows": p.n_baseline_windows,
                "n_scored_windows": p.n_scored_windows,
                "n_samples": p.n_samples,
                "n_positive": p.n_positive,
                "positive_ratio": p.positive_ratio,
                "window_step_seconds": p.window_step_seconds,
                "horizon_windows": p.horizon_windows,
                "trigger_bounds": p.trigger_bounds,
                "trigger_lower": p.trigger_lower,
                "trigger_upper": p.trigger_upper,
                "trigger_width": p.trigger_width,
            }
            for p in participant_summaries
        ],
        "skipped_participants": [
            {"participant_id": pid, "reason": reason}
            for pid, reason in skipped_participants
        ],
        "xgboost_params": model.get_params(),
    }

    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {metadata_path}")

    latest_model = output_dir / "latest_future_trigger_classifier.json"
    latest_metadata = output_dir / "latest_future_trigger_classifier_metadata.json"

    if latest_model.is_symlink() or latest_model.exists():
        latest_model.unlink()
    if latest_metadata.is_symlink() or latest_metadata.exists():
        latest_metadata.unlink()

    shutil.copy2(model_path, latest_model)
    shutil.copy2(metadata_path, latest_metadata)

    return model_path


def main() -> None:
    """Main training pipeline for future trigger classification."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost binary classifier for future reactive feedback triggering"
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
    parser.add_argument(
        "--consecutive-outside-windows",
        type=int,
        default=DEFAULT_CONSECUTIVE_OUTSIDE_WINDOWS,
        help="Consecutive observed windows outside calibrated bounds required for label=1.",
    )
    parser.add_argument(
        "--baseline-duration-seconds",
        type=float,
        default=DEFAULT_BASELINE_DURATION_SECONDS,
        help="Baseline duration used to calibrate participant trigger bounds.",
    )
    parser.add_argument(
        "--prediction-horizon-seconds",
        type=float,
        default=DEFAULT_PREDICTION_HORIZON_SECONDS,
        help="How many seconds ahead the classifier should predict trigger state.",
    )
    parser.add_argument(
        "--history-window-size",
        type=int,
        default=None,
        help="Optional override for history window size.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("XGBoost Future Reactive Trigger Classifier Training")
    print("=" * 70)

    if args.config:
        print(f"Loading config from: {args.config}")
        system_config = SystemConfig.from_file(args.config)
        training_config = system_config.training
        reactive_config = system_config.reactive_tool
        baseline_duration_seconds = float(
            args.baseline_duration_seconds
            if args.baseline_duration_seconds is not None
            else getattr(system_config.controller, "calibration_duration_seconds", DEFAULT_BASELINE_DURATION_SECONDS)
        )
    else:
        print("Using default configs")
        training_config = TrainingConfig()
        reactive_config = ReactiveToolConfig()
        baseline_duration_seconds = float(args.baseline_duration_seconds)

    if args.history_window_size is not None:
        training_config.history_window_size = int(args.history_window_size)

    history_size = int(training_config.history_window_size)
    if history_size < 1:
        raise ValueError("history_window_size must be >= 1")

    if args.consecutive_outside_windows < 1:
        raise ValueError("consecutive_outside_windows must be >= 1")

    if args.prediction_horizon_seconds <= 0:
        raise ValueError("prediction_horizon_seconds must be > 0")

    if args.split_dir:
        split_dir = Path(args.split_dir)
    else:
        base_dir = Path(__file__).parent.parent
        split_dir = base_dir / "data" / "processed" / "splits"

    effective_data_path = (
        Path(training_config.data_path)
        if getattr(training_config, "data_path", None) is not None
        else None
    )

    input_columns = _get_input_columns(training_config)

    print("Loading processed dataset...")
    df = load_processed_data(effective_data_path)
    print(f"Loaded {len(df)} windows from {df['participant_id'].nunique()} participants")

    df = _filter_observed_windows(df)
    print(f"Observed windows retained: {len(df)}")

    _validate_columns_exist(df, ["participant_id", "trial", "window_start", "window_end"], "required")
    _validate_columns_exist(df, input_columns, "input")

    print("\nBuilding future-trigger classification labels and samples...")
    built = build_future_trigger_classification_dataset(
        df=df,
        input_columns=input_columns,
        history_size=history_size,
        prediction_horizon_seconds=float(args.prediction_horizon_seconds),
        reactive_config=reactive_config,
        baseline_duration_seconds=baseline_duration_seconds,
        consecutive_outside_windows=int(args.consecutive_outside_windows),
    )

    if len(built.X) == 0:
        raise RuntimeError("No classifier samples were produced. Check baseline/windows availability.")

    print(f"Built {len(built.X)} samples from {len(set(built.participant_ids))} participants")
    print(f"Skipped participants: {len(built.skipped_participants)}")

    if built.participant_summaries:
        sorted_by_width = sorted(
            [p for p in built.participant_summaries if p.trigger_width is not None],
            key=lambda p: p.trigger_width
        )
        sorted_by_pos_ratio = sorted(
            built.participant_summaries,
            key=lambda p: p.positive_ratio,
            reverse=True
        )

        print("\nParticipants with smallest trigger width:")
        for p in sorted_by_width[:10]:
            print(
                f"  pid={p.participant_id} "
                f"trigger_width={p.trigger_width:.4f} "
                f"pos_ratio={p.positive_ratio:.4f} "
                f"baseline_windows={p.n_baseline_windows}"
            )

        print("\nParticipants with highest positive ratio:")
        for p in sorted_by_pos_ratio[:10]:
            print(
                f"  pid={p.participant_id} "
                f"pos_ratio={p.positive_ratio:.4f} "
                f"trigger_width={p.trigger_width if p.trigger_width is not None else 'None'} "
                f"baseline_windows={p.n_baseline_windows}"
            )

    overall_balance = _class_balance(built.y)
    print(
        "Class balance (overall): "
        f"pos={overall_balance['n_positive']} / {overall_balance['n_samples']} "
        f"({overall_balance['positive_ratio']:.4f})"
    )

    print("\nSplitting dataset by participant...")
    splits = split_by_participant(
        built.X,
        built.y,
        built.participant_ids,
        config=training_config,
        split_dir=split_dir,
    )

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print("\nFinal dataset sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    train_balance = _class_balance(y_train)
    val_balance = _class_balance(y_val)
    test_balance = _class_balance(y_test)

    print(
        "Class balance (train/val/test positive ratio): "
        f"{train_balance['positive_ratio']:.4f} / "
        f"{val_balance['positive_ratio']:.4f} / "
        f"{test_balance['positive_ratio']:.4f}"
    )

    if train_balance["n_positive"] == 0 or train_balance["n_negative"] == 0:
        raise RuntimeError(
            "Training split has only one class. Adjust data/split settings before training classifier."
        )

    scale_pos_weight = _compute_scale_pos_weight(y_train)
    if train_balance["positive_ratio"] < 0.1 or train_balance["positive_ratio"] > 0.9:
        print(
            "[WARN] Labels are highly imbalanced in train split. "
            f"Using scale_pos_weight={scale_pos_weight:.4f}"
        )

    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=training_config,
        scale_pos_weight=scale_pos_weight,
    )

    metrics = evaluate_classifier(
        model=model,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    get_feature_importance(model, built.feature_names)

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "models" / "trained"

    class_balance = {
        "overall": overall_balance,
        "train": train_balance,
        "val": val_balance,
        "test": test_balance,
    }

    save_model_and_metadata(
        model=model,
        feature_names=built.feature_names,
        input_columns=built.input_columns,
        output_dir=output_dir,
        history_window_size=history_size,
        prediction_horizon_seconds=float(args.prediction_horizon_seconds),
        consecutive_outside_windows=int(args.consecutive_outside_windows),
        baseline_duration_seconds=baseline_duration_seconds,
        reactive_config=reactive_config,
        metrics=metrics,
        class_balance=class_balance,
        participant_summaries=built.participant_summaries,
        skipped_participants=built.skipped_participants,
    )

    print("\n" + "=" * 70)
    print("Future trigger-classifier training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()