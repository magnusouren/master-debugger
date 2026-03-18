"""Quick line-chart plotting for experiment logs.

Usage:
    python -m backend.analysis.plot_experiments --show

It reads CSV logs from logs/experiments, plots timestamp vs. score, colors by
operation mode, and overlays predicted scores as a dashed line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# Default location of experiment logs relative to the repo root
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "experiments"


def _parse_estimates(csv_path: Path) -> pd.DataFrame:
    """Load a single experiment CSV into a tidy DataFrame of estimates."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "data" not in df.columns or "mode" not in df.columns:
        raise ValueError(f"File {csv_path} is missing expected columns")

    estimates: list[dict] = []
    for _, row in df.iterrows():
        if str(row.get("event_type", "")) != "user_state_estimate_logged":
            continue

        try:
            payload = json.loads(row["data"])
        except json.JSONDecodeError:
            continue

        score = payload.get("score")
        if score is None:
            continue

        # Prefer payload timestamp; if it's a numeric epoch (seconds), convert accordingly.
        ts_val = payload.get("timestamp")
        if isinstance(ts_val, (int, float)):
            ts = pd.to_datetime(ts_val, unit="s", utc=True).tz_convert(None)
        else:
            ts = pd.to_datetime(ts_val or row["timestamp"])
        mode = str(row.get("mode", "")).upper() or "UNKNOWN"
        source_type = payload.get("source_type") or payload.get("metadata", {}).get("source_type")
        forecast_id = payload.get("forecast_id") or payload.get("metadata", {}).get("forecast_id")
        is_predicted = source_type == "predicted_features" or forecast_id not in (None, "", "null")

        estimates.append(
            {
                "timestamp": ts,
                "score": float(score),
                "mode": mode,
                "is_predicted": bool(is_predicted),
            }
        )

    tidy = pd.DataFrame(estimates)
    if tidy.empty:
        return tidy

    return tidy.sort_values("timestamp")


def _color_for_mode(mode: str) -> Optional[str]:
    palette = {
        "CONTROL": "#1f77b4",
        "REACTIVE": "#ff7f0e",
        "PROACTIVE": "#2ca02c",
    }
    return palette.get(mode.upper())


def plot_estimates(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    actual = df[~df["is_predicted"]]
    for mode, group in actual.groupby("mode"):
        group_sorted = group.sort_values("timestamp")
        ax.plot(
            group_sorted["timestamp"],
            group_sorted["score"],
            label=f"{mode.title()} observed",
            color=_color_for_mode(mode),
            linewidth=1.8,
        )

    predicted = df[df["is_predicted"]].sort_values("timestamp")
    if not predicted.empty:
        ax.plot(
            predicted["timestamp"],
            predicted["score"],
            label="predicted",
            linestyle="--",
            linewidth=1.5,
            color="#444444",
        )

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _select_files(
    log_dir: Path, pattern: str, session_id: Optional[str], files: Optional[Iterable[str]], include_all: bool
) -> List[Path]:
    if files:
        resolved = [Path(f).expanduser().resolve() for f in files]
        return resolved

    candidates = sorted(log_dir.glob(pattern))
    if session_id:
        candidates = [c for c in candidates if session_id in c.name]

    if not candidates:
        raise FileNotFoundError(f"No experiment CSVs found in {log_dir} matching '{pattern}'")

    if include_all:
        return candidates

    return [max(candidates, key=lambda p: p.stat().st_mtime)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment scores as a line chart.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory containing experiment CSV logs")
    parser.add_argument("--pattern", default="experiment_*.csv", help="Glob pattern for log files")
    parser.add_argument("--session-id", help="Substring to match a specific session id (filters filenames)")
    parser.add_argument("--file", action="append", help="Explicit CSV file(s) to plot")
    parser.add_argument("--all", dest="include_all", action="store_true", help="Plot all matching files instead of the newest")
    parser.add_argument("--output-dir", help="Directory to save plots (defaults to backend/analysis/figures)")
    parser.add_argument("--show", action="store_true", help="Display the plot window")

    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(__file__).resolve().parent / "figures"

    csv_files = _select_files(log_dir, args.pattern, args.session_id, args.file, args.include_all)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[plt.Figure] = []
    for csv_path in csv_files:
        df = _parse_estimates(csv_path)
        if df.empty:
            continue
        fig = plot_estimates(df, title=csv_path.stem)
        figures.append(fig)
        output_path = output_dir / f"{csv_path.stem}.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved {output_path}")

    if args.show and figures:
        plt.show()


if __name__ == "__main__":
    main()
