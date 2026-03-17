"""
Plot the logged user-state scores from an experiment CSV in logs/experiments.

Usage:
    python -m backend.analysis.plot_load_score path/to/logs/experiments/<session>.csv [--output out.png] [--show]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Mode-specific colors for the line segments
MODE_COLORS: Dict[str, str] = {
    "control": "#1f77b4",
    "reactive": "#d62728",
    "proactive": "#ff7f0e",
    "questionnaire": "#9467bd",
    "unknown": "#7f7f7f",
}


def _parse_timestamp(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def _load_points(csv_path: Path) -> List[Tuple[float, float, str]]:
    """Load (time_sec, score, mode) tuples from an experiment log CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    # Establish a base timestamp for fallback time calculation
    base_ts: datetime | None = None
    for ts in df["timestamp"].dropna().tolist():
        try:
            base_ts = _parse_timestamp(str(ts))
            break
        except Exception:
            continue

    points: List[Tuple[float, float, str]] = []

    for _, row in df.iterrows():
        if row.get("event_type") != "user_state_estimate_logged":
            continue

        raw_data = row.get("data")
        try:
            payload = json.loads(raw_data) if isinstance(raw_data, str) else {}
        except json.JSONDecodeError:
            continue

        score = payload.get("score")
        if score is None:
            continue

        time_sec = payload.get("seconds_since_start")
        if time_sec is None and base_ts is not None:
            ts_str = row.get("timestamp")
            try:
                ts = _parse_timestamp(str(ts_str))
                time_sec = (ts - base_ts).total_seconds()
            except Exception:
                continue

        if time_sec is None:
            continue

        mode = str(row.get("mode") or "unknown").lower()
        points.append((float(time_sec), float(score), mode))

    points.sort(key=lambda p: p[0])
    return points


def _plot(points: List[Tuple[float, float, str]], output_path: Path | None, show: bool) -> None:
    if not points:
        raise ValueError("No user_state_estimate_logged rows found in the CSV.")

    # Break into contiguous segments per mode so colors switch when the mode switches.
    segments: List[Tuple[str, List[float], List[float]]] = []
    current_mode: str | None = None
    times: List[float] = []
    scores: List[float] = []

    for t, s, mode in points:
        if current_mode is None:
            current_mode = mode
        if mode != current_mode:
            segments.append((current_mode, times, scores))
            current_mode = mode
            times, scores = [], []
        times.append(t)
        scores.append(s)

    if times and scores and current_mode is not None:
        segments.append((current_mode, times, scores))

    fig, ax = plt.subplots(figsize=(10, 5))
    used_labels = set()

    for mode, seg_times, seg_scores in segments:
        color = MODE_COLORS.get(mode, MODE_COLORS["unknown"])
        label = mode if mode not in used_labels else "_nolegend_"
        ax.plot(seg_times, seg_scores, color=color, label=label)
        used_labels.add(mode)

    ax.set_xlabel("Seconds since start")
    ax.set_ylabel("Load score")
    ax.set_title("User load score over time")
    if used_labels:
        ax.legend(title="Mode")
    ax.grid(True, linestyle="--", alpha=0.4)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    if show or not output_path:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot load scores (user_state_estimate_logged) from experiment logs.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to logs/experiments/<session>.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (PNG). Defaults to <csv_path>.score.png",
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

    points = _load_points(csv_path)

    output_path = Path(args.output) if args.output else csv_path.with_suffix(".score.png")
    _plot(points, output_path=output_path, show=args.show)

    if output_path:
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
