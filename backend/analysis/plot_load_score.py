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

from backend.layers.reactive_tool import ReactiveTool, ReactiveToolConfig
from backend.services.logger_service import LoggerService
from backend.types import PredictedFeatures, WindowFeatures


@dataclass
class SeriesPoint:
    timestamp: datetime
    score: float

@dataclass
class FeatureEvent:
    timestamp: float
    window: WindowFeatures


DEFAULT_ENABLED_METRICS = [
    "pupil_diameter",
    "fixation_duration",
    "saccade_amplitude",
    "ipi",
    "data_quality",
]


def _extract_valid_ratio(payload: dict, features: dict) -> float:
    return float(
        payload.get("avg_valid_sample_ratio")
        or features.get("dq_valid_ratio_any")
        or features.get("pupil_valid_ratio")
        or 0.0
    )


def _parse_row(row: pd.Series) -> Optional[FeatureEvent]:
    """Parse a CSV row into a WindowFeatures event for reactive scoring."""
    data_field = row.get("data")
    try:
        payload = json.loads(data_field) if isinstance(data_field, str) else {}
    except json.JSONDecodeError:
        return None

    event_type = row.get("event_type")

    if event_type == "observed_feature_window_logged":
        features = payload.get("features", {}) or {}
        window_start = payload.get("window_start")
        window_end = payload.get("window_end") or window_start
        if window_start is None or window_end is None:
            return None
        wf = WindowFeatures(
            window_start=float(window_start),
            window_end=float(window_end),
            window_id=payload.get("window_id"),
            is_predicted=False,
            forecast_id=None,
            features=features,
            enabled_metrics=list(DEFAULT_ENABLED_METRICS),
            sample_count=int(features.get("pupil_window_sample_count", 0) or 0),
            valid_sample_ratio=_extract_valid_ratio(payload, features),
        )
        ts = float(window_end)
        return FeatureEvent(timestamp=ts, window=wf)

    if event_type == "predicted_feature_window_logged":
        features = payload.get("predicted_features") or payload.get("features") or {}
        target_start = payload.get("target_window_start")
        target_end = payload.get("target_window_end") or target_start
        target_time = payload.get("target_time") or target_end or target_start
        if target_start is None or target_end is None or target_time is None:
            return None
        pf = PredictedFeatures(
            prediction_timestamp=float(payload.get("prediction_timestamp") or target_time),
            target_window_start=float(target_start),
            target_window_end=float(target_end),
            horizon_seconds=float(payload.get("prediction_horizon_seconds") or 0.0),
            forecast_id=payload.get("forecast_id"),
            window_id=payload.get("window_id"),
            features=features,
            enabled_metrics=list(DEFAULT_ENABLED_METRICS),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
        )
        wf = pf.to_window_features()
        wf.valid_sample_ratio = _extract_valid_ratio(payload, features)
        ts = float(target_time)
        return FeatureEvent(timestamp=ts, window=wf)

    return None


def _load_series(path: Path) -> Tuple[List[SeriesPoint], List[SeriesPoint]]:
    df = pd.read_csv(path)

    events: List[FeatureEvent] = []
    for _, row in df.iterrows():
        evt = _parse_row(row)
        if evt:
            events.append(evt)

    events.sort(key=lambda e: e.timestamp)

    logger = LoggerService(
        experiment_level="ERROR",
        system_level="ERROR",
        features_level="ERROR",
    )
    rt = ReactiveTool(config=ReactiveToolConfig(), logger=logger)
    rt.start()

    observed: List[SeriesPoint] = []
    predicted: List[SeriesPoint] = []

    for evt in events:
        rt.add_features(evt.window)
        estimate = rt.estimate()
        if not estimate:
            continue
        point = SeriesPoint(
            timestamp=datetime.fromtimestamp(evt.timestamp),
            score=float(estimate.score.score),
        )
        if evt.window.is_predicted:
            predicted.append(point)
        else:
            observed.append(point)

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
