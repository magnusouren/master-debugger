"""
Generate a load score line chart from a feature log CSV.

Usage:
    python -m backend.analysis.plot_load_score path/to/logs/features_<session>.csv [--output out.png] [--show]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

from backend.models.forecast_feature_schema import compute_score_from_target_components


@dataclass
class SeriesPoint:
    timestamp: datetime
    score: float


def _parse_row(row: pd.Series) -> Tuple[Optional[SeriesPoint], Optional[SeriesPoint]]:
    """Parse a single CSV row into observed/predicted points."""
    data_field = row.get("data")
    try:
        payload = json.loads(data_field) if isinstance(data_field, str) else {}
    except json.JSONDecodeError:
        return None, None

    event_type = row.get("event_type")

    if event_type == "observed_feature_window_logged":
        features = payload.get("features", {}) or {}
        ts = payload.get("window_start") or payload.get("window_end")
        if ts is None:
            return None, None
        score = compute_score_from_target_components(features)
        return SeriesPoint(datetime.fromtimestamp(float(ts)), float(score)), None

    if event_type == "predicted_feature_window_logged":
        features = payload.get("predicted_features") or payload.get("features") or {}
        ts = (
            payload.get("target_time")
            or payload.get("target_window_start")
            or payload.get("target_window_end")
        )
        if ts is None:
            return None, None
        score = compute_score_from_target_components(features)
        return None, SeriesPoint(datetime.fromtimestamp(float(ts)), float(score))

    return None, None


def _load_series(path: Path) -> Tuple[List[SeriesPoint], List[SeriesPoint]]:
    df = pd.read_csv(path)

    observed: List[SeriesPoint] = []
    predicted: List[SeriesPoint] = []

    for _, row in df.iterrows():
        obs_point, pred_point = _parse_row(row)
        if obs_point:
            observed.append(obs_point)
        if pred_point:
            predicted.append(pred_point)

    observed.sort(key=lambda p: p.timestamp)
    predicted.sort(key=lambda p: p.timestamp)
    return observed, predicted


def _plot_series(
    observed: Iterable[SeriesPoint],
    predicted: Iterable[SeriesPoint],
    output_path: Optional[Path],
    show: bool,
) -> None:
    obs_times = [p.timestamp for p in observed]
    obs_scores = [p.score for p in observed]
    pred_times = [p.timestamp for p in predicted]
    pred_scores = [p.score for p in predicted]

    fig, ax = plt.subplots(figsize=(10, 5))
    if obs_times:
        ax.plot(obs_times, obs_scores, label="Observed score", color="#1f77b4")
    if pred_times:
        ax.plot(pred_times, pred_scores, label="Predicted score", color="#d62728")

    ax.set_xlabel("Time")
    ax.set_ylabel("Load score")
    ax.set_title("User load score over time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    if show or not output_path:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot load score over time from feature logs.")
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to logs/features/<session>.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (PNG). Defaults to <csv_path>_score.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (also saves if --output is set).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    observed, predicted = _load_series(csv_path)
    if not observed and not predicted:
        raise ValueError("No observed or predicted feature rows found in the CSV.")

    output_path = Path(args.output) if args.output else csv_path.with_suffix(".score.png")

    _plot_series(observed, predicted, output_path=output_path, show=args.show)
    if output_path:
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
