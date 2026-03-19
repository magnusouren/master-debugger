"""Quick line-chart plotting for experiment logs.

Usage:
    python -m backend.analysis.plot_experiments --show

It reads CSV logs from backend/logs/experiments, plots timestamp vs. score,
colors by operation mode, and overlays predicted scores.
Top panel shows both raw score and EMA/smoothed score when available.
If available, it also plots the five contributor values over time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# Default location of experiment logs (backend/logs/experiments)
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "experiments"

# Deterministic color cache for modes (covers known modes and any extras like QUESTIONNAIRE)
_PALETTE_BASE = {
    "CONTROL": "#1f77b4",
    "REACTIVE": "#ff7f0e",
    "PROACTIVE": "#2ca02c",
    "QUESTIONNAIRE": "#9467bd",
}
_MODE_COLOR_CACHE: dict[str, str] = {}
_CONTRIB_KEYS = [
    "contrib_ipa",
    "contrib_fixation_duration",
    "contrib_anticipation",
    "contrib_perceived_difficulty",
    "contrib_ipi",
]
_CONTRIB_LABELS = {
    "contrib_ipa": "IPA",
    "contrib_fixation_duration": "Fixation",
    "contrib_anticipation": "Anticipation",
    "contrib_perceived_difficulty": "Perceived difficulty",
    "contrib_ipi": "IPI",
}
_CONTRIB_COLORS = {
    "contrib_ipa": "#1f77b4",
    "contrib_fixation_duration": "#ff7f0e",
    "contrib_anticipation": "#2ca02c",
    "contrib_perceived_difficulty": "#d62728",
    "contrib_ipi": "#9467bd",
}


def _to_float_or_none(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_estimates(csv_path: Path) -> pd.DataFrame:
    """Load a single experiment CSV into a tidy DataFrame of estimates."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "data" not in df.columns or "mode" not in df.columns:
        raise ValueError(f"File {csv_path} is missing expected columns")

    # Detect baseline calibration end so we can separate baseline-phase points in plots.
    baseline_end_ts: Optional[pd.Timestamp] = None
    baseline_rows = df[df.get("event_type", "").astype(str) == "baseline_calibration_completed"]
    if not baseline_rows.empty:
        # Use the latest baseline completion timestamp in this file.
        baseline_end_ts = pd.to_datetime(baseline_rows["timestamp"]).max()

    estimates: list[dict] = []
    for _, row in df.iterrows():
        event_type = str(row.get("event_type", ""))
        if event_type not in ("user_state_estimate_logged", "observer_user_state_estimate_logged"):
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
        metadata = payload.get("metadata", {}) or {}
        source_type = payload.get("source_type") or metadata.get("source_type")
        is_predicted = source_type == "predicted_features"
        contributing_features = payload.get("contributing_features", {}) or {}
        score_ema = _to_float_or_none(score)
        if score_ema is None:
            continue
        score_raw = _to_float_or_none(metadata.get("raw_score"))
        if score_raw is None:
            score_raw = score_ema

        contrib_ipa = _to_float_or_none(
            contributing_features.get("contrib_ipa", contributing_features.get("ipa_score"))
        )
        contrib_fixation = _to_float_or_none(
            contributing_features.get(
                "contrib_fixation_duration",
                contributing_features.get("fixation_duration_score"),
            )
        )
        contrib_anticipation = _to_float_or_none(
            contributing_features.get("contrib_anticipation", contributing_features.get("anticipation_score"))
        )
        contrib_perceived = _to_float_or_none(
            contributing_features.get(
                "contrib_perceived_difficulty",
                contributing_features.get("perceived_difficulty_score"),
            )
        )
        contrib_ipi = _to_float_or_none(
            contributing_features.get("contrib_ipi", contributing_features.get("ipi_score"))
        )

        estimates.append(
            {
                "timestamp": ts,
                "score": float(score_ema),
                "score_ema": float(score_ema),
                "score_raw": float(score_raw),
                "mode": mode,
                "event_type": event_type,
                "is_predicted": bool(is_predicted),
                "is_baseline_period": bool(baseline_end_ts is not None and ts <= baseline_end_ts),
                "contrib_ipa": contrib_ipa,
                "contrib_fixation_duration": contrib_fixation,
                "contrib_anticipation": contrib_anticipation,
                "contrib_perceived_difficulty": contrib_perceived,
                "contrib_ipi": contrib_ipi,
            }
        )

    tidy = pd.DataFrame(estimates)
    if tidy.empty:
        return tidy

    return tidy.sort_values("timestamp")


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


def plot_estimates(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, (ax_score, ax_contrib) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4]},
    )

    predicted = df[df["is_predicted"]].sort_values("timestamp")
    actual_all = df[~df["is_predicted"]].sort_values("timestamp")

    # For proactive comparison, prioritize non-baseline observed streams (e.g., observer estimates).
    if not predicted.empty:
        observer_actual = actual_all[
            actual_all["event_type"] == "observer_user_state_estimate_logged"
        ]
        if not observer_actual.empty:
            actual = observer_actual.sort_values("timestamp")
        else:
            actual = actual_all[~actual_all["is_baseline_period"]].sort_values("timestamp")
    else:
        actual = actual_all

    # Plot observed segments without bridging gaps between separated occurrences of the same mode.
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
        label = None if mode in used_labels else f"{mode.title()} observed (EMA)"
        ax_score.plot(
            seg_sorted["timestamp"],
            seg_sorted["score_ema"],
            label=label,
            color=_color_for_mode(mode),
            linewidth=1.8,
        )
        raw_label = None if mode in used_labels else f"{mode.title()} observed (raw)"
        ax_score.plot(
            seg_sorted["timestamp"],
            seg_sorted["score_raw"],
            label=raw_label,
            color=_color_for_mode(mode),
            linewidth=1.2,
            linestyle=":",
            alpha=0.75,
        )
        used_labels.add(mode)

    if not predicted.empty:
        ax_score.plot(
            predicted["timestamp"],
            predicted["score_ema"],
            label="predicted (EMA)",
            linestyle="--",
            linewidth=1.5,
            color="#444444",
        )
        ax_score.plot(
            predicted["timestamp"],
            predicted["score_raw"],
            label="predicted (raw)",
            linestyle="-.",
            linewidth=1.2,
            color="#111111",
            alpha=0.85,
        )
        first_pred_ts = predicted["timestamp"].min()
        ax_score.axvline(
            first_pred_ts,
            color="#666666",
            linewidth=1.0,
            linestyle=":",
            alpha=0.8,
        )
        ax_score.text(
            first_pred_ts,
            ax_score.get_ylim()[1],
            " first prediction",
            va="top",
            ha="left",
            fontsize=8,
            color="#666666",
        )
        if actual.empty:
            ax_score.text(
                0.01,
                0.03,
                "No observed proactive stream in this session (only baseline observed).",
                transform=ax_score.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#666666",
            )

    ax_score.set_title(title)
    ax_score.set_ylabel("Score")
    ax_score.grid(True, linestyle=":", linewidth=0.8)
    ax_score.legend(loc="upper right")

    contrib_df = df.copy()
    has_any_contrib = False
    for key in _CONTRIB_KEYS:
        if key not in contrib_df.columns:
            continue
        series = contrib_df[key].dropna()
        if series.empty:
            continue
        has_any_contrib = True
        observed_source = actual if not actual.empty else contrib_df.iloc[0:0]
        observed_series = observed_source[["timestamp", key]].dropna() if key in observed_source.columns else observed_source.iloc[0:0]
        predicted_series = contrib_df[contrib_df["is_predicted"]][["timestamp", key]].dropna()

        if not observed_series.empty:
            ax_contrib.plot(
                observed_series["timestamp"],
                observed_series[key],
                color=_CONTRIB_COLORS[key],
                linewidth=1.2,
                label=f"{_CONTRIB_LABELS[key]} observed",
            )
        if not predicted_series.empty:
            ax_contrib.plot(
                predicted_series["timestamp"],
                predicted_series[key],
                color=_CONTRIB_COLORS[key],
                linewidth=1.2,
                linestyle="--",
                label=f"{_CONTRIB_LABELS[key]} predicted",
            )

    ax_contrib.set_xlabel("Timestamp")
    ax_contrib.set_ylabel("Contributor (0-1)")
    ax_contrib.set_ylim(-0.02, 1.02)
    ax_contrib.grid(True, linestyle=":", linewidth=0.8)
    if has_any_contrib:
        ax_contrib.legend(loc="upper right", ncol=2, fontsize=8)
    else:
        ax_contrib.text(
            0.5,
            0.5,
            "No contributor values found in this log",
            ha="center",
            va="center",
            transform=ax_contrib.transAxes,
        )

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
