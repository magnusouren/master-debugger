"""
Dataset preparation for XGBoost forecasting.

Creates training samples with:
- X: History window of contributor values (typically 120 windows = 60 seconds)
- y: Future component values used by ReactiveTool scoring

Reuses ReactiveTool._estimate_rule_based() for scoring.
Splits data by participant for proper evaluation.

Usage:
    from backend.training.dataset import prepare_dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json

from backend.types import WindowFeatures, ReactiveToolConfig, TrainingConfig
from backend.layers.reactive_tool import ReactiveTool
from backend.models.forecast_feature_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    compute_contributor_features,
)
from backend.services.logger_service import LoggerService

# Defaults aligned with TrainingConfig: 60s history and ~30s horizon at 2 Hz.
HISTORY_WINDOW_SIZE = 120
PREDICTION_HORIZON = 60

# Suppress per-sample ReactiveTool system logs during offline dataset generation.
TRAINING_LOGGER = LoggerService(experiment_level="ERROR", system_level="ERROR")


def load_processed_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load preprocessed EMIP features.
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


def row_to_window_features(row: pd.Series) -> WindowFeatures:
    """Convert a DataFrame row to WindowFeatures object."""
    # Keep all raw metrics in WindowFeatures so rule-based target scoring
    # can read the fields it expects (e.g., pupil_ipa, ipi_value, velocities).
    features = {}
    for col, val in row.items():
        if isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
            features[col] = float(val)

    return WindowFeatures(
        window_start=row.get('window_start', 0),
        window_end=row.get('window_end', 0),
        features=features,
        sample_count=int(row.get('sample_count', 0)),
        valid_sample_ratio=float(row.get('valid_ratio', 0)),
        # Match SignalProcessingConfig defaults used in main.
        enabled_metrics=[
            'pupil_diameter',
            'fixation_duration',
            'saccade_amplitude',
            'blink_rate',
            'data_quality',
            'ipi',
        ],
    )


def compute_cognitive_load_score(windows: List[WindowFeatures]) -> float:
    """
    Compute cognitive load score using ReactiveTool's rule-based method.

    This reuses the exact same logic used at runtime.
    """
    # Use default config - enabled_metrics comes from WindowFeatures, not config
    config = ReactiveToolConfig()
    tool = ReactiveTool(config=config, logger=TRAINING_LOGGER)

    # Use the internal method directly
    return tool._estimate_rule_based(windows)


def create_sequences(
    df: pd.DataFrame,
    history_size: int = HISTORY_WINDOW_SIZE,
    horizon: int = PREDICTION_HORIZON
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create input-output sequences for training.

    Args:
        df: DataFrame with features
        history_size: Number of past windows to use as input
        horizon: Number of windows to look ahead for target

    Returns:
        X: Input features (N, history_size * n_features)
        y: Target component vectors (N, 5)
        participant_ids: List of participant IDs for each sample
    """
    X_list = []
    y_list = []
    participant_ids = []

    # Process each participant/trial separately
    for (pid, trial), group in df.groupby(['participant_id', 'trial']):
        group = group.sort_values('window_start').reset_index(drop=True)

        # Need at least history_size + horizon windows
        if len(group) < history_size + horizon:
            continue

        # Create sequences
        for i in range(len(group) - history_size - horizon + 1):
            # History window rows
            history_rows = group.iloc[i:i + history_size]

            # Target window(s)
            target_rows = group.iloc[i + history_size:i + history_size + horizon]

            # Build target components from target horizon window(s)
            # using the same raw metrics expected by ReactiveTool.
            ipa_series = pd.to_numeric(target_rows.get("pupil_ipa"), errors="coerce").dropna()
            fixation_series = pd.to_numeric(
                target_rows.get("fixation_mean_duration_ms"), errors="coerce"
            ).dropna()
            velocity_series = pd.to_numeric(
                target_rows.get("saccade_mean_velocity"), errors="coerce"
            ).dropna()
            ipi_series = pd.to_numeric(target_rows.get("ipi_value"), errors="coerce").dropna()

            if (
                len(ipa_series) == 0
                or len(fixation_series) == 0
                or len(velocity_series) == 0
                or len(ipi_series) == 0
            ):
                continue

            target_vector = [
                float(ipa_series.mean()),
                float(fixation_series.mean()),
                float(velocity_series.mean()),
                float(velocity_series.std(ddof=0)) if len(velocity_series) >= 2 else 0.0,
                float(ipi_series.mean()),
            ]

            # Flatten history features
            # Exactly 5 calculated contributor values are used as model inputs.
            features = []
            for _, row in history_rows.iterrows():
                contribs = compute_contributor_features(row.to_dict())
                for col in FEATURE_COLUMNS:
                    features.append(float(contribs[col]))

            X_list.append(features)
            y_list.append(target_vector)
            participant_ids.append(pid)

    return np.array(X_list), np.array(y_list), participant_ids


def split_by_participant(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: List[str],
    config: Optional[TrainingConfig] = None,
    split_dir: Optional[Path] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data by participant (not by sample).
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
            # Shuffle and split participants once, then persist to disk.
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
        # In-memory split only (legacy behavior).
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

    # Split samples
    train_mask = np.array([p in train_participants for p in participant_ids])
    val_mask = np.array([p in val_participants for p in participant_ids])
    test_mask = np.array([p in test_participants for p in participant_ids])

    return {
        'train': (X[train_mask], y[train_mask]),
        'val': (X[val_mask], y[val_mask]),
        'test': (X[test_mask], y[test_mask]),
    }


def get_feature_names(history_size: int = HISTORY_WINDOW_SIZE) -> List[str]:
    """Get feature names for the flattened input vector."""
    names = []
    for t in range(history_size):
        for col in FEATURE_COLUMNS:
            names.append(f"t-{history_size - t - 1}_{col}")
    return names


def prepare_dataset(
    config: Optional[TrainingConfig] = None,
    data_path: Optional[Path] = None,
    split_dir: Optional[Path] = None,
) -> Dict[str, any]:
    """
    Complete dataset preparation pipeline.

    Args:
        config: TrainingConfig with hyperparameters (uses defaults if None)
        data_path: Optional path to processed data (overrides config.data_path)
        split_dir: Optional folder for persistent participant split files.
    """
    if config is None:
        config = TrainingConfig()

    if split_dir is None:
        base_dir = Path(__file__).parent.parent
        split_dir = base_dir / "data" / "processed" / "splits"

    # Use provided data_path or fall back to config
    effective_data_path = data_path
    if effective_data_path is None and config.data_path is not None:
        effective_data_path = Path(config.data_path)

    print("Loading processed data...")
    df = load_processed_data(effective_data_path)
    print(f"Loaded {len(df)} windows from {df['participant_id'].nunique()} participants")

    history_size = config.history_window_size
    horizon = config.prediction_horizon

    print(f"\nCreating sequences (history={history_size}, horizon={horizon})...")
    X, y, participant_ids = create_sequences(df, history_size, horizon)
    print(f"Created {len(X)} samples")

    print("\nSplitting by participant...")
    splits = split_by_participant(X, y, participant_ids, config=config, split_dir=split_dir)

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(splits['train'][0])} samples")
    print(f"  Val: {len(splits['val'][0])} samples")
    print(f"  Test: {len(splits['test'][0])} samples")

    return {
        'X_train': splits['train'][0],
        'y_train': splits['train'][1],
        'X_val': splits['val'][0],
        'y_val': splits['val'][1],
        'X_test': splits['test'][0],
        'y_test': splits['test'][1],
        'feature_names': get_feature_names(config.history_window_size),
        'config': config,
    }


if __name__ == "__main__":
    # Use default TrainingConfig
    training_config = TrainingConfig()
    dataset = prepare_dataset(config=training_config)
    print(f"\nFeatures per sample: {dataset['X_train'].shape[1]}")
    print(f"Target mean: {dataset['y_train'].mean():.3f}")
    print(f"Target std: {dataset['y_train'].std():.3f}")
