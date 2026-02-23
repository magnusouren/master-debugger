"""
Dataset preparation for XGBoost forecasting.

Creates training samples with:
- X: History window of metrics (e.g., last 5 windows)
- y: Future cognitive load score (e.g., score at t+k)

Reuses ReactiveTool._estimate_rule_based() for scoring.
Splits data by participant for proper evaluation.

Usage:
    from backend.training.dataset import prepare_dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from backend.types import WindowFeatures, ReactiveToolConfig, TrainingConfig
from backend.layers.reactive_tool import ReactiveTool


# Feature columns from SignalProcessingLayer output
FEATURE_COLUMNS = [
    'pupil_mean',
    'pupil_std',
    'pupil_slope',
    'pupil_range',
    'pupil_mean_abs_vel',
    'fixation_count',
    'fixation_mean_duration_ms',
    'saccade_count',
    'saccade_mean_amplitude',
    'saccade_mean_velocity',
    'saccade_velocity_std',
    'gaze_disp_total',
]

# Number of historical windows to use as input
HISTORY_WINDOW_SIZE = 5

# Number of windows to look ahead for target
PREDICTION_HORIZON = 2


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
    features = {}
    for col in FEATURE_COLUMNS:
        val = row.get(col)
        if pd.notna(val):
            features[col] = float(val)

    return WindowFeatures(
        window_start=row.get('window_start', 0),
        window_end=row.get('window_end', 0),
        features=features,
        sample_count=int(row.get('sample_count', 0)),
        valid_sample_ratio=float(row.get('valid_ratio', 0)),
        enabled_metrics=['pupil_diameter', 'fixation_duration', 'saccade_amplitude', 'gaze_dispersion'],
    )


def compute_cognitive_load_score(windows: List[WindowFeatures]) -> float:
    """
    Compute cognitive load score using ReactiveTool's rule-based method.

    This reuses the exact same logic used at runtime.
    """
    config = ReactiveToolConfig(
        enabled_metrics=['pupil_diameter', 'fixation_duration', 'saccade_amplitude', 'gaze_dispersion'],
    )
    tool = ReactiveTool(config=config)

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
        y: Target cognitive load scores (N,)
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

            # Convert to WindowFeatures and compute target score
            target_windows = [row_to_window_features(row) for _, row in target_rows.iterrows()]
            target_score = compute_cognitive_load_score(target_windows)

            # Flatten history features
            features = []
            for _, row in history_rows.iterrows():
                for col in FEATURE_COLUMNS:
                    val = row.get(col)
                    features.append(float(val) if pd.notna(val) else 0.0)

            X_list.append(features)
            y_list.append(target_score)
            participant_ids.append(pid)

    return np.array(X_list), np.array(y_list), participant_ids


def split_by_participant(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: List[str],
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data by participant (not by sample).
    """
    if config is None:
        config = TrainingConfig()

    unique_participants = list(set(participant_ids))
    n_participants = len(unique_participants)

    print(f"Total participants: {n_participants}")

    # Shuffle and split participants
    np.random.seed(config.random_state)
    np.random.shuffle(unique_participants)

    n_train = int(n_participants * config.train_ratio)
    n_val = int(n_participants * config.val_ratio)

    train_participants = set(unique_participants[:n_train])
    val_participants = set(unique_participants[n_train:n_train + n_val])
    test_participants = set(unique_participants[n_train + n_val:])

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
) -> Dict[str, any]:
    """
    Complete dataset preparation pipeline.

    Args:
        config: TrainingConfig with hyperparameters (uses defaults if None)
        data_path: Optional path to processed data (overrides config.data_path)
    """
    if config is None:
        config = TrainingConfig()

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
    splits = split_by_participant(X, y, participant_ids, config=config)

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
