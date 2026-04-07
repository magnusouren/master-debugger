"""
Dataset preparation for XGBoost forecasting.

Creates sequence-to-one training samples with:
- X: Flattened history of WindowFeatures from the last N windows
     + engineered summary features computed from that history
- y: One future WindowFeatures row at a fixed horizon

This is intended for forecasting future WindowFeatures directly:
    history of WindowFeatures -> one future WindowFeatures

The dataset is split by participant for proper evaluation.

Usage:
    from backend.training.dataset import prepare_dataset
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.types import TrainingConfig, WindowFeatures

# Defaults aligned with preprocess/runtime
HISTORY_WINDOW_SIZE = 60
PREDICTION_HORIZON = 20

DEFAULT_INPUT_COLUMNS = [
    "pupil_ipa",
    "fixation_mean_duration_ms",
    "blink_rate_per_min",
    "saccade_count",
]

DEFAULT_TARGET_COLUMNS = list(DEFAULT_INPUT_COLUMNS)  # Predict same features by default

DEFAULT_ENABLED_METRICS = [
    "fixation_duration",
    "saccade_amplitude",
    "pupil_diameter",
    "blink_rate",
    "data_quality",
    "ipi",
]


@dataclass
class ColumnNormStats:
    """Per-participant z-score statistics for one feature column."""
    mean: float
    std: float
    sample_count: int


def load_processed_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load preprocessed EMIP window features.
    """
    if data_path is None:
        base_dir = Path(__file__).parent.parent
        data_path = base_dir / "data" / "processed" / "emip_features.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_path}\n"
            "Run 'python -m backend.training.preprocess' first."
        )

    return pd.read_parquet(data_path)


def _to_float(value: Any) -> Optional[float]:
    """
    Safely coerce a scalar value to finite float.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(f):
        return None
    return f


def _get_input_columns(config: Optional[TrainingConfig] = None) -> List[str]:
    """
    Resolve input columns from config if present, otherwise defaults.
    """
    if config is not None:
        cols = getattr(config, "input_feature_columns", None)
        if cols:
            return list(cols)
    return list(DEFAULT_INPUT_COLUMNS)


def _get_target_columns(config: Optional[TrainingConfig] = None) -> List[str]:
    """
    Resolve target columns from config if present, otherwise defaults.
    """
    if config is not None:
        cols = getattr(config, "target_feature_columns", None)
        if cols:
            return list(cols)
    return list(DEFAULT_TARGET_COLUMNS)


def _validate_columns_exist(df: pd.DataFrame, columns: List[str], label: str) -> None:
    """
    Raise a clear error if expected columns are missing from the dataframe.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {label} columns in processed dataframe: {missing}\n"
            f"Available columns are: {list(df.columns)}"
        )


def _extract_row_values(row: pd.Series, columns: List[str]) -> Optional[List[float]]:
    """
    Extract a list of numeric values from a row.

    Returns None if any required value is missing/non-numeric/non-finite.
    """
    values: List[float] = []

    for col in columns:
        val = _to_float(row.get(col))
        if val is None:
            return None
        values.append(val)

    return values


def fit_participant_normalizers(
    df: pd.DataFrame,
    columns: List[str],
    participant_column: str = "participant_id",
) -> Dict[str, Dict[str, ColumnNormStats]]:
    """
    Fit simple per-participant z-score statistics for the requested columns.

    Missing / non-finite values are ignored. Zero or invalid std falls back to 1.0.
    """
    normalizers: Dict[str, Dict[str, ColumnNormStats]] = {}

    for participant_id, group in df.groupby(participant_column):
        participant_key = str(participant_id)
        participant_stats: Dict[str, ColumnNormStats] = {}

        for col in columns:
            numeric = pd.to_numeric(group[col], errors="coerce")
            valid = numeric.replace([np.inf, -np.inf], np.nan).dropna()

            if valid.empty:
                participant_stats[col] = ColumnNormStats(mean=0.0, std=1.0, sample_count=0)
                continue

            mean_val = float(valid.mean())
            std_val = float(valid.std(ddof=0))
            if not np.isfinite(std_val) or std_val <= 0.0:
                std_val = 1.0

            participant_stats[col] = ColumnNormStats(
                mean=mean_val,
                std=std_val,
                sample_count=int(valid.shape[0]),
            )

        normalizers[participant_key] = participant_stats

    return normalizers


def apply_participant_normalization(
    df: pd.DataFrame,
    participant_normalizers: Dict[str, Dict[str, ColumnNormStats]],
    columns: List[str],
    participant_column: str = "participant_id",
) -> pd.DataFrame:
    """
    Apply per-participant z-score normalization to the requested columns only.

    Missing values remain missing.
    """
    normalized_df = df.copy()

    # Normalize only model-space columns, but make sure they can hold z-scored
    # floating-point values before we assign back into the dataframe.
    for col in columns:
        normalized_df[col] = pd.to_numeric(normalized_df[col], errors="coerce").astype("float64")

    for participant_id, participant_stats in participant_normalizers.items():
        participant_mask = normalized_df[participant_column].astype(str) == participant_id
        if not participant_mask.any():
            continue

        for col in columns:
            stats = participant_stats.get(col)
            if stats is None:
                continue

            numeric = pd.to_numeric(normalized_df.loc[participant_mask, col], errors="coerce")
            normalized_df.loc[participant_mask, col] = (numeric - stats.mean) / stats.std

    return normalized_df


def serialize_participant_normalizers(
    participant_normalizers: Dict[str, Dict[str, ColumnNormStats]],
) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    """Convert fitted participant normalizers to plain metadata-safe dictionaries."""
    serialized: Dict[str, Dict[str, Dict[str, float | int]]] = {}

    for participant_id, participant_stats in participant_normalizers.items():
        serialized[participant_id] = {
            col: {
                "mean": float(stats.mean),
                "std": float(stats.std),
                "sample_count": int(stats.sample_count),
            }
            for col, stats in participant_stats.items()
        }

    return serialized


def row_to_window_features(
    row: pd.Series,
    enabled_metrics: Optional[List[str]] = None,
) -> WindowFeatures:
    """
    Convert a processed dataframe row back into a WindowFeatures object.
    Useful later when reconstructing forecasted rows into runtime format.
    """
    features: Dict[str, float] = {}

    for col, val in row.items():
        f = _to_float(val)
        if f is not None:
            features[col] = f

    sample_count_val = _to_float(row.get("sample_count", row.get("dq_sample_count", 0)))
    valid_ratio_val = _to_float(row.get("valid_ratio", row.get("dq_valid_ratio_any", 0.0)))

    return WindowFeatures(
        window_start=float(row.get("window_start", 0.0)),
        window_end=float(row.get("window_end", 0.0)),
        features=features,
        sample_count=int(sample_count_val) if sample_count_val is not None else 0,
        valid_sample_ratio=float(valid_ratio_val) if valid_ratio_val is not None else 0.0,
        enabled_metrics=list(enabled_metrics or DEFAULT_ENABLED_METRICS),
    )


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def _safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if arr.size > 0 else 0.0


def _safe_min(arr: np.ndarray) -> float:
    return float(np.min(arr)) if arr.size > 0 else 0.0


def _safe_max(arr: np.ndarray) -> float:
    return float(np.max(arr)) if arr.size > 0 else 0.0


def _safe_range(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.max(arr) - np.min(arr))


def _safe_slope(arr: np.ndarray) -> float:
    """
    Linear slope over equally spaced time steps.
    """
    if arr.size < 2:
        return 0.0

    x = np.arange(arr.size, dtype=np.float32)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(arr))

    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return 0.0

    numer = float(np.sum((x - x_mean) * (arr - y_mean)))
    return numer / denom


def _build_summary_features(
    history_matrix: np.ndarray,
    input_columns: List[str],
) -> Tuple[List[float], List[str]]:
    """
    Build engineered summary features from the history window.

    Args:
        history_matrix: shape (history_size, n_features)
        input_columns: feature names for each column in history_matrix

    Returns:
        values, names
    """
    values: List[float] = []
    names: List[str] = []

    for col_idx, col_name in enumerate(input_columns):
        series = history_matrix[:, col_idx]

        last_1 = series[-1:]
        last_3 = series[-3:]
        last_5 = series[-5:]
        last_10 = series[-10:]
        prev_5 = series[-10:-5] if series.shape[0] >= 10 else np.array([], dtype=series.dtype)

        feature_map = {
            f"{col_name}__last": float(series[-1]),
            f"{col_name}__mean_last_3": _safe_mean(last_3),
            f"{col_name}__mean_last_5": _safe_mean(last_5),
            f"{col_name}__mean_last_10": _safe_mean(last_10),
            f"{col_name}__std_last_5": _safe_std(last_5),
            f"{col_name}__std_last_10": _safe_std(last_10),
            f"{col_name}__min_last_10": _safe_min(last_10),
            f"{col_name}__max_last_10": _safe_max(last_10),
            f"{col_name}__range_last_10": _safe_range(last_10),
            f"{col_name}__diff_last_1": float(series[-1] - series[-2]) if series.shape[0] >= 2 else 0.0,
            f"{col_name}__diff_mean_5_vs_prev_5": (
                _safe_mean(last_5) - _safe_mean(prev_5)
                if prev_5.size > 0
                else 0.0
            ),
            f"{col_name}__slope_last_5": _safe_slope(last_5),
            f"{col_name}__slope_last_10": _safe_slope(last_10),
        }

        for name, value in feature_map.items():
            names.append(name)
            values.append(float(value))

    return values, names


def _build_input_vector(
    history_rows: pd.DataFrame,
    input_columns: List[str],
) -> Optional[List[float]]:
    """
    Build the final input vector:
    - flattened raw history
    - engineered summary features
    """
    history_values: List[List[float]] = []

    for _, row in history_rows.iterrows():
        row_values = _extract_row_values(row, input_columns)
        if row_values is None:
            return None
        history_values.append(row_values)

    history_matrix = np.asarray(history_values, dtype=np.float32)
    flattened_history = history_matrix.flatten().tolist()

    summary_values, _ = _build_summary_features(history_matrix, input_columns)

    return flattened_history + summary_values


def create_sequences(
    df: pd.DataFrame,
    history_size: int = HISTORY_WINDOW_SIZE,
    horizon_steps: int = PREDICTION_HORIZON,
    input_columns: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sequence-to-one forecasting samples.

    Each sample is:
    - X: history_size past windows + summary features from that history
    - y: one future window at horizon_steps ahead

    Args:
        df: Processed dataframe with window-based features
        history_size: Number of past windows to use as input
        horizon_steps: Number of future windows ahead to predict
        input_columns: Columns to use from each history row
        target_columns: Columns to predict from the target row

    Returns:
        X: shape (N, history_features)
        y: shape (N, len(target_columns))
        participant_ids: participant id for each sample
    """
    if input_columns is None:
        input_columns = list(DEFAULT_INPUT_COLUMNS)
    if target_columns is None:
        target_columns = list(DEFAULT_TARGET_COLUMNS)

    X_list: List[List[float]] = []
    y_list: List[List[float]] = []
    participant_ids: List[str] = []

    for (pid, trial), group in df.groupby(["participant_id", "trial"]):
        group = group.sort_values("window_start").reset_index(drop=True)

        min_required = history_size + horizon_steps
        if len(group) < min_required:
            continue

        n_samples = len(group) - history_size - horizon_steps + 1

        for i in range(n_samples):
            history_rows = group.iloc[i:i + history_size]

            target_idx = i + history_size - 1 + horizon_steps
            current_idx = i + history_size - 1

            current_row = group.iloc[current_idx]
            target_row = group.iloc[target_idx]

            current_vector = _extract_row_values(current_row, target_columns)
            future_vector = _extract_row_values(target_row, target_columns)

            if current_vector is None or future_vector is None:
                continue

            target_vector = [
                future_val - current_val
                for current_val, future_val in zip(current_vector, future_vector)
            ]

            input_vector = _build_input_vector(history_rows, input_columns)
            if input_vector is None:
                continue

            X_list.append(input_vector)
            y_list.append(target_vector)
            participant_ids.append(str(pid))

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        participant_ids,
    )


def get_feature_names(
    history_size: int = HISTORY_WINDOW_SIZE,
    input_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Get names for flattened input features + summary features.
    """
    if input_columns is None:
        input_columns = list(DEFAULT_INPUT_COLUMNS)

    names: List[str] = []

    # Raw flattened history
    for t in range(history_size):
        for col in input_columns:
            names.append(f"t-{history_size - t - 1}_{col}")

    # Summary features
    dummy_history = np.zeros((max(history_size, 10), len(input_columns)), dtype=np.float32)
    _, summary_names = _build_summary_features(dummy_history, input_columns)
    names.extend(summary_names)

    return names


def split_by_participant(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: List[str],
    config: Optional[TrainingConfig] = None,
    split_dir: Optional[Path] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data by participant, not by sample.
    """
    if config is None:
        config = TrainingConfig()

    unique_participants = sorted(set(participant_ids))
    n_participants = len(unique_participants)

    print(f"Total participants: {n_participants}")

    train_participants: set[str]
    val_participants: set[str]
    test_participants: set[str]

    if split_dir is not None:
        split_dir.mkdir(parents=True, exist_ok=True)
        train_file = split_dir / "train_participants.json"
        val_file = split_dir / "val_participants.json"
        test_file = split_dir / "test_participants.json"

        split_files_exist = train_file.exists() and val_file.exists() and test_file.exists()

        if split_files_exist:
            train_participants = set(json.loads(train_file.read_text()))
            val_participants = set(json.loads(val_file.read_text()))
            test_participants = set(json.loads(test_file.read_text()))
            print(f"Loaded participant split from: {split_dir}")
        else:
            shuffled = unique_participants.copy()
            np.random.seed(config.random_state)
            np.random.shuffle(shuffled)

            n_train = int(n_participants * config.train_ratio)
            n_val = int(n_participants * config.val_ratio)

            train_participants = set(shuffled[:n_train])
            val_participants = set(shuffled[n_train:n_train + n_val])
            test_participants = set(shuffled[n_train + n_val:])

            train_file.write_text(json.dumps(sorted(train_participants), indent=2))
            val_file.write_text(json.dumps(sorted(val_participants), indent=2))
            test_file.write_text(json.dumps(sorted(test_participants), indent=2))
            print(f"Saved participant split to: {split_dir}")
    else:
        shuffled = unique_participants.copy()
        np.random.seed(config.random_state)
        np.random.shuffle(shuffled)

        n_train = int(n_participants * config.train_ratio)
        n_val = int(n_participants * config.val_ratio)

        train_participants = set(shuffled[:n_train])
        val_participants = set(shuffled[n_train:n_train + n_val])
        test_participants = set(shuffled[n_train + n_val:])

    print(f"Train participants: {len(train_participants)}")
    print(f"Val participants: {len(val_participants)}")
    print(f"Test participants: {len(test_participants)}")

    train_mask = np.array([p in train_participants for p in participant_ids])
    val_mask = np.array([p in val_participants for p in participant_ids])
    test_mask = np.array([p in test_participants for p in participant_ids])

    return {
        "train": (X[train_mask], y[train_mask]),
        "val": (X[val_mask], y[val_mask]),
        "test": (X[test_mask], y[test_mask]),
    }


def prepare_dataset(
    config: Optional[TrainingConfig] = None,
    data_path: Optional[Path] = None,
    split_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Complete dataset preparation pipeline.
    """
    if config is None:
        config = TrainingConfig()

    if split_dir is None:
        base_dir = Path(__file__).parent.parent
        split_dir = base_dir / "data" / "processed" / "splits"

    effective_data_path = data_path
    if effective_data_path is None and config.data_path is not None:
        effective_data_path = Path(config.data_path)

    input_columns = _get_input_columns(config)
    target_columns = _get_target_columns(config)

    print("Loading processed data...")
    df = load_processed_data(effective_data_path)
    print(f"Loaded {len(df)} windows from {df['participant_id'].nunique()} participants")

    _validate_columns_exist(df, input_columns, "input")
    _validate_columns_exist(df, target_columns, "target")

    normalization_columns = sorted(set(input_columns + target_columns))
    participant_normalizers = fit_participant_normalizers(
        df=df,
        columns=normalization_columns,
    )
    normalized_df = apply_participant_normalization(
        df=df,
        participant_normalizers=participant_normalizers,
        columns=normalization_columns,
    )

    history_size = int(getattr(config, "history_window_size", HISTORY_WINDOW_SIZE))
    horizon_steps = int(getattr(config, "prediction_horizon", PREDICTION_HORIZON))

    print("\nMissing-value summary:")
    for col in set(input_columns + target_columns):
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"{col}: {missing} missing of {len(df)}")

    print(
        f"\nCreating sequence-to-one samples "
        f"(history={history_size}, horizon_steps={horizon_steps})..."
    )
    X, y, participant_ids = create_sequences(
        df=normalized_df,
        history_size=history_size,
        horizon_steps=horizon_steps,
        input_columns=input_columns,
        target_columns=target_columns,
    )
    print(f"Created {len(X)} samples")

    print("\nSplitting by participant...")
    splits = split_by_participant(X, y, participant_ids, config=config, split_dir=split_dir)

    print("\nFinal dataset sizes:")
    print(f"  Train: {len(splits['train'][0])} samples")
    print(f"  Val:   {len(splits['val'][0])} samples")
    print(f"  Test:  {len(splits['test'][0])} samples")

    return {
        "X_train": splits["train"][0],
        "y_train": splits["train"][1],
        "X_val": splits["val"][0],
        "y_val": splits["val"][1],
        "X_test": splits["test"][0],
        "y_test": splits["test"][1],
        "feature_names": get_feature_names(history_size, input_columns=input_columns),
        "input_columns": input_columns,
        "target_columns": target_columns,
        "normalization_mode": "participant_zscore",
        "target_type": "delta",
        "participant_normalizers": participant_normalizers,
        "participant_normalizers_serialized": serialize_participant_normalizers(participant_normalizers),
        "config": config,
    }


if __name__ == "__main__":
    training_config = TrainingConfig()
    dataset = prepare_dataset(config=training_config)

    print(f"\nFeatures per sample: {dataset['X_train'].shape[1]}")
    print(f"Targets per sample:  {dataset['y_train'].shape[1]}")

    print("\nTarget statistics (train split):")
    for idx, col in enumerate(dataset["target_columns"]):
        series = dataset["y_train"][:, idx]
        print(
            f"  {col:<24} "
            f"mean={series.mean():>8.4f} "
            f"std={series.std():>8.4f} "
            f"min={series.min():>8.4f} "
            f"max={series.max():>8.4f}"
        )
