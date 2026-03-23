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
from typing import Any, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# Default location of experiment logs relative to the repo root
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "experiments"

# Deterministic color cache for modes (covers known modes and any extras like QUESTIONNAIRE)
_PALETTE_BASE = {
    "CONTROL": "#1f77b4",
    "REACTIVE": "#ff7f0e",
    "PROACTIVE": "#2ca02c",
    "QUESTIONNAIRE": "#9467bd",
}
_MODE_COLOR_CACHE: dict[str, str] = {}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_estimates(csv_path: Path) -> tuple[pd.DataFrame, Optional[dict[str, float | str]]]:
    """Load a single experiment CSV into a tidy DataFrame of estimates + trigger bounds."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "data" not in df.columns or "mode" not in df.columns:
        raise ValueError(f"File {csv_path} is missing expected columns")

    estimates: list[dict] = []
    trigger_bounds: Optional[dict[str, float | str]] = None
    for _, row in df.iterrows():
        event_type = str(row.get("event_type", ""))

        try:
            payload = json.loads(row["data"])
        except json.JSONDecodeError:
            continue

        # Prefer explicit runtime trigger bounds when available.
        if event_type == "feedback_delivery_threshold_met":
            lower = _to_float(payload.get("lower_bound"))
            upper = _to_float(payload.get("upper_bound"))
            rule = str(payload.get("rule", ""))
            if lower is not None and upper is not None:
                trigger_bounds = {
                    "rule": rule or "baseline_mean_pm_2sd",
                    "lower": lower,
                    "upper": upper,
                }
                mean = _to_float(payload.get("baseline_mean"))
                std = _to_float(payload.get("baseline_std"))
                if mean is not None:
                    trigger_bounds["mean"] = mean
                if std is not None:
                    trigger_bounds["std"] = std
            else:
                threshold = _to_float(payload.get("threshold"))
                if rule == "static_min_score" and threshold is not None:
                    trigger_bounds = {
                        "rule": rule,
                        "threshold": threshold,
                    }

        # Fallback: infer bounds from baseline summary if no explicit trigger event is present yet.
        if trigger_bounds is None and event_type == "baseline_calibration_completed":
            score_metric = payload.get("metrics", {}).get("cognitive_load_score", {})
            mean = _to_float(score_metric.get("mean"))
            std = _to_float(score_metric.get("std"))
            p02_5 = _to_float(score_metric.get("p02_5"))
            p97_5 = _to_float(score_metric.get("p97_5"))
            if p02_5 is not None and p97_5 is not None:
                trigger_bounds = {
                    "rule": "baseline_empirical_p2_5_p97_5",
                    "lower": p02_5,
                    "upper": p97_5,
                }
                if mean is not None:
                    trigger_bounds["mean"] = mean
                if std is not None:
                    trigger_bounds["std"] = std
            elif mean is not None and std is not None:
                trigger_bounds = {
                    "rule": "baseline_mean_pm_2sd",
                    "lower": mean - (2.0 * std),
                    "upper": mean + (2.0 * std),
                    "mean": mean,
                    "std": std,
                }

        if event_type not in ("user_state_estimate_logged", "observer_user_state_estimate_logged"):
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
        is_predicted = source_type == "predicted_features"

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
        return tidy, trigger_bounds

    return tidy.sort_values("timestamp"), trigger_bounds


def _color_for_mode(mode: str) -> Optional[str]:
    key = mode.upper()
    if key in _MODE_COLOR_CACHE:
        return _MODE_COLOR_CACHE[key]

    if key in _PALETTE_BASE:
        _MODE_COLOR_CACHE[key] = _PALETTE_BASE[key]
        return _MODE_COLOR_CACHE[key]

    # Assign deterministic fallback colors from tab10, cycling by hash to stay stable across runs
    tab10 = plt.get_cmap("tab10").colors
    color = tab10[hash(key) % len(tab10)]
    _MODE_COLOR_CACHE[key] = color
    return color

def _smooth_series(
    series: pd.Series,
    method: str = "ema",
    window: int = 5,
    alpha: float = 0.3,
) -> pd.Series:
    """Smooth a score series for plotting."""
    if series.empty:
        return series

    if method == "rolling":
        return series.rolling(window=window, min_periods=1, center=True).mean()

    if method == "ema":
        return series.ewm(alpha=alpha, adjust=False).mean()

    return series


def plot_estimates(df: pd.DataFrame, title: str, trigger_bounds: Optional[dict[str, float | str]] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot observed segments without bridging gaps between separated occurrences of the same mode.
    actual = df[~df["is_predicted"]].sort_values("timestamp")
    segments: list[tuple[str, pd.DataFrame]] = []
    current_mode: Optional[str] = None
    current_rows: list[dict] = []

    for _, row in actual.iterrows():
        row_mode = row["mode"]
        if current_mode is None or row_mode == current_mode:
            current_rows.append(row)
            current_mode = row_mode
        else:
            if current_rows:
                segments.append((current_mode, pd.DataFrame(current_rows)))
            current_rows = [row]
            current_mode = row_mode
    if current_rows:
        segments.append((current_mode, pd.DataFrame(current_rows)))

    used_labels: set[str] = set()
    for mode, segment in segments:
        seg_sorted = segment.sort_values("timestamp")
        label = None if mode in used_labels else f"{mode.title()} observed"
        smoothed_score = _smooth_series(seg_sorted["score"], method="ema", alpha=0.25)

        ax.plot(
            seg_sorted["timestamp"],
            smoothed_score,
            label=label,
            color=_color_for_mode(mode),
            linewidth=1.8,
        )
        used_labels.add(mode)

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

    if trigger_bounds:
        rule = str(trigger_bounds.get("rule", ""))
        lower = _to_float(trigger_bounds.get("lower"))
        upper = _to_float(trigger_bounds.get("upper"))
        threshold = _to_float(trigger_bounds.get("threshold"))
        mean = _to_float(trigger_bounds.get("mean"))

        if lower is not None and upper is not None:
            band_label = (
                "Trigger band (2.5–97.5 pct)"
                if rule == "baseline_empirical_p2_5_p97_5"
                else "Trigger band (mean ± 2 SD)"
            )
            ax.axhspan(lower, upper, color="#999999", alpha=0.1, label=band_label, zorder=0)
            ax.axhline(lower, color="#b22222", linestyle=":", linewidth=1.2, label=f"Lower bound ({lower:.3f})")
            ax.axhline(upper, color="#b22222", linestyle=":", linewidth=1.2, label=f"Upper bound ({upper:.3f})")
            if mean is not None:
                ax.axhline(mean, color="#6a6a6a", linestyle="-.", linewidth=1.1, label=f"Baseline mean ({mean:.3f})")
        elif threshold is not None:
            ax.axhline(
                threshold,
                color="#b22222",
                linestyle=":",
                linewidth=1.2,
                label=f"Trigger threshold ({threshold:.3f})",
            )

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
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
        df, trigger_bounds = _parse_estimates(csv_path)
        if df.empty:
            continue
        fig = plot_estimates(df, title=csv_path.stem, trigger_bounds=trigger_bounds)
        figures.append(fig)
        output_path = output_dir / f"{csv_path.stem}.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved {output_path}")

    if args.show and figures:
        plt.show()


if __name__ == "__main__":
    main()
