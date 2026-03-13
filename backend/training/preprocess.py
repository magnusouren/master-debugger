"""
Preprocess EMIP dataset for XGBoost training.

This script:
1. Parses EMIP TSV files → GazeSample objects
2. Uses SignalProcessingLayer to compute WindowFeatures
3. Saves processed data for training

Reuses existing metric calculation logic from signal_processing.py.

Usage:
    python -m backend.training.preprocess
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.types import GazeSample, WindowFeatures, SignalProcessingConfig
from backend.layers.signal_processing import SignalProcessingLayer
from backend.services.logger_service import LoggerService


# Screen resolution for EMIP dataset (for normalizing gaze coordinates)
EMIP_SCREEN_WIDTH = 1920
EMIP_SCREEN_HEIGHT = 1200

# Processing settings
WINDOW_LENGTH_SECONDS = 1.0
OUTPUT_FREQUENCY_HZ = 2.0  # 2 windows per second with 50% overlap

# Keep preprocessing console output readable by suppressing frequent
# per-window warning logs from SignalProcessingLayer.
PREPROCESS_LOGGER = LoggerService(experiment_level="ERROR", system_level="ERROR")


def parse_emip_file(filepath: Path) -> pd.DataFrame:
    """
    Parse an EMIP TSV file, skipping header comments.
    """
    # Read file, skip comment lines starting with ##
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find header line (first non-comment line)
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.startswith('##'):
            header_idx = i
            break

    # Read as TSV from header line onwards
    df = pd.read_csv(
        filepath,
        sep='\t',
        skiprows=header_idx,
        low_memory=False
    )

    return df


def emip_row_to_gaze_sample(row: pd.Series) -> Optional[GazeSample]:
    """
    Convert an EMIP data row to a GazeSample object.
    """
    try:
        timestamp = row.get('Time', 0)
        if pd.isna(timestamp) or timestamp == 0:
            return None

        # Convert timestamp from microseconds to seconds
        timestamp_sec = timestamp / 1_000_000.0

        # Get validity flags
        l_valid = row.get('L Validity', 0) == 1
        r_valid = row.get('R Validity', 0) == 1

        # Get gaze coordinates (normalize to [0, 1])
        l_x = row.get('L POR X [px]')
        l_y = row.get('L POR Y [px]')
        r_x = row.get('R POR X [px]')
        r_y = row.get('R POR Y [px]')

        # Normalize coordinates
        if pd.notna(l_x) and l_x != 0:
            l_x = l_x / EMIP_SCREEN_WIDTH
        else:
            l_x = None

        if pd.notna(l_y) and l_y != 0:
            l_y = l_y / EMIP_SCREEN_HEIGHT
        else:
            l_y = None

        if pd.notna(r_x) and r_x != 0:
            r_x = r_x / EMIP_SCREEN_WIDTH
        else:
            r_x = None

        if pd.notna(r_y) and r_y != 0:
            r_y = r_y / EMIP_SCREEN_HEIGHT
        else:
            r_y = None

        # Get pupil diameters (already in mm)
        l_pupil = row.get('L Mapped Diameter [mm]')
        r_pupil = row.get('R Mapped Diameter [mm]')

        if pd.isna(l_pupil) or l_pupil == 0:
            l_pupil = None
        if pd.isna(r_pupil) or r_pupil == 0:
            r_pupil = None

        return GazeSample(
            timestamp=timestamp_sec,
            left_eye_x=l_x,
            left_eye_y=l_y,
            right_eye_x=r_x,
            right_eye_y=r_y,
            left_pupil_diameter=l_pupil,
            right_pupil_diameter=r_pupil,
            left_eye_valid=l_valid and l_x is not None,
            right_eye_valid=r_valid and r_x is not None,
        )
    except Exception as e:
        return None


def process_participant(filepath: Path) -> List[Dict[str, Any]]:
    """
    Process a single participant's data file using SignalProcessingLayer.

    Returns list of window feature dictionaries.
    """
    participant_id = filepath.stem.split('_')[0]
    print(f"  Processing participant {participant_id}...")

    try:
        df = parse_emip_file(filepath)
    except Exception as e:
        print(f"    Error parsing file: {e}")
        return []

    if 'Time' not in df.columns:
        print(f"    Missing Time column")
        return []

    all_windows: List[Dict[str, Any]] = []

    # Process each trial separately
    trials = df['Trial'].unique() if 'Trial' in df.columns else [1]

    for trial in trials:
        trial_df = df[df['Trial'] == trial] if 'Trial' in df.columns else df

        if len(trial_df) < 100:  # Need minimum samples
            continue

        # Convert to GazeSample objects
        samples: List[GazeSample] = []
        for _, row in trial_df.iterrows():
            sample = emip_row_to_gaze_sample(row)
            if sample is not None:
                samples.append(sample)

        if len(samples) < 50:
            continue

        # Create SignalProcessingLayer with default config (120 Hz to match Tobii tracker)
        # Only override settings that differ from defaults
        config = SignalProcessingConfig(
            window_length_seconds=WINDOW_LENGTH_SECONDS,
            output_frequency_hz=OUTPUT_FREQUENCY_HZ,
            min_valid_sample_ratio=0.3,
            require_both_eyes_valid=False,
        )

        processor = SignalProcessingLayer(config=config, logger=PREPROCESS_LOGGER)

        # Collect windows via callback
        windows_from_trial: List[WindowFeatures] = []

        def collect_window(wf: WindowFeatures):
            windows_from_trial.append(wf)

        processor.register_output_callback(collect_window)
        processor.start()

        # Feed samples in batches
        batch_size = 100
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            processor.add_samples(batch)

        processor.stop()

        # Convert WindowFeatures to dicts with metadata
        for wf in windows_from_trial:
            if wf.valid_sample_ratio < 0.3:
                continue

            record = {
                'participant_id': participant_id,
                'trial': int(trial),
                'window_start': wf.window_start,
                'window_end': wf.window_end,
                'valid_ratio': wf.valid_sample_ratio,
                'sample_count': wf.sample_count,
            }
            # Add all computed features
            record.update(wf.features)

            all_windows.append(record)

    print(f"    Generated {len(all_windows)} windows")
    return all_windows


def process_all_participants(data_dir: Path) -> pd.DataFrame:
    """Process all participant files and return combined DataFrame."""
    rawdata_dir = data_dir / "rawdata"

    if not rawdata_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {rawdata_dir}")

    all_windows: List[Dict[str, Any]] = []

    files = sorted(rawdata_dir.glob("*_rawdata.tsv"))
    print(f"Found {len(files)} participant files")

    for filepath in files:
        windows = process_participant(filepath)
        all_windows.extend(windows)

    print(f"\nTotal windows: {len(all_windows)}")

    return pd.DataFrame(all_windows)


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "emip_dataset"
    output_dir = base_dir / "data" / "processed"

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all participants
    df = process_all_participants(data_dir)

    if len(df) == 0:
        print("ERROR: No windows generated")
        return

    # Save to parquet
    output_path = output_dir / "emip_features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")

    # Also save as CSV for inspection
    csv_path = output_dir / "emip_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV copy to: {csv_path}")

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Total windows: {len(df)}")
    print(f"Features: {list(df.columns)}")


if __name__ == "__main__":
    main()
