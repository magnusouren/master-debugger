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

# Fallback bounds for logs (e.g., replay runs) without baseline/threshold events.
_DEFAULT_TRIGGER_BOUNDS: dict[str, float | str] = {
    "rule": "fallback_default_band",
    "lower": 0.45,
    "upper": 0.55,
    "mean": 0.5,
    "std": 0.05,
}

_FEEDBACK_INTERACTION_MARKERS: dict[str, str] = {
    "feedback_presented_to_user": "P",
    "feedback_accepted_by_user": "A",
    "feedback_rejected_by_user": "R",
    "feedback_marked_done_by_user": "D",
    "feedback_dismissed_by_user": "X",
    "feedback_highlighted_in_code": "H",
}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """Parse mixed timestamp values from log payload/columns."""
    try:
        if isinstance(value, (int, float)):
            numeric = float(value)
            # Runtime payloads can contain relative seconds (e.g. 8939.12),
            # which should not be interpreted as Unix epoch.
            if numeric >= 1.0e12:
                ts_num = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
            elif numeric >= 1.0e9:
                ts_num = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
            else:
                return None
            if pd.isna(ts_num):
                return None
            return ts_num.tz_convert(None)
        ts = pd.to_datetime(value, utc=True)
        if pd.isna(ts):
            return None
        return ts.tz_convert(None)
    except (TypeError, ValueError):
        return None


def _marker_for_feedback_interaction(event_type: str, payload: dict[str, Any]) -> Optional[str]:
    marker = _FEEDBACK_INTERACTION_MARKERS.get(event_type)
    if marker:
        return marker

    # Fallback: derive from action_taken when event_type is generic.
    action_taken = str(payload.get("action_taken", "")).strip().lower()
    action_map = {
        "presented": "P",
        "accepted": "A",
        "rejected": "R",
        "done": "D",
        "dismissed": "X",
        "highlighted": "H",
    }
    if action_taken in action_map:
        return action_map[action_taken]

    if event_type.startswith("feedback_interaction_unknown_type") and ":" in event_type:
        unknown = event_type.split(":", 1)[1].strip()
        if unknown:
            return unknown[0].upper()

    return None


def _trigger_bounds_from_payload(payload: dict[str, Any]) -> Optional[dict[str, float | str]]:
    """Extract trigger bounds from payload if lower/upper are available."""
    lower = _to_float(payload.get("lower"))
    upper = _to_float(payload.get("upper"))
    if lower is None or upper is None:
        return None

    out: dict[str, float | str] = {
        "rule": str(payload.get("rule", "baseline_mean_pm_2sd")),
        "lower": lower,
        "upper": upper,
    }
    mean = _to_float(payload.get("mean"))
    std = _to_float(payload.get("std"))
    if mean is not None:
        out["mean"] = mean
    if std is not None:
        out["std"] = std
    return out


def _event_timestamp(row: pd.Series) -> Optional[pd.Timestamp]:
    """Use CSV log-row timestamp for event timing in plots."""
    return _parse_timestamp(row.get("timestamp"))


def _parse_estimates(csv_path: Path) -> tuple[pd.DataFrame, Optional[dict[str, float | str]], pd.DataFrame]:
    """Load a single experiment CSV into estimates, trigger bounds, and feedback interactions."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "data" not in df.columns or "mode" not in df.columns:
        raise ValueError(f"File {csv_path} is missing expected columns")

    estimates: list[dict] = []
    feedback_interactions: list[dict[str, Any]] = []
    trigger_bounds: Optional[dict[str, float | str]] = None
    baseline_completed_ts: Optional[pd.Timestamp] = None
    for _, row in df.iterrows():
        event_type = str(row.get("event_type", ""))

        try:
            payload = json.loads(row["data"])
        except json.JSONDecodeError:
            continue

        marker = _marker_for_feedback_interaction(event_type, payload)
        if marker is not None:
            interaction_ts = _parse_timestamp(payload.get("timestamp"))
            if interaction_ts is None:
                interaction_ts = _parse_timestamp(row.get("timestamp"))
            if interaction_ts is not None and not pd.isna(interaction_ts):
                feedback_interactions.append(
                    {
                        "timestamp": interaction_ts,
                        "marker": marker,
                        "event_type": event_type,
                    }
                )

        # Priority 1: explicit calibrated trigger bounds (post-baseline, runtime scoring space).
        if event_type == "feedback_trigger_bounds_calibrated":
            parsed_bounds = _trigger_bounds_from_payload(payload)
            if parsed_bounds is not None:
                bounds_ts = _event_timestamp(row)
                if bounds_ts is not None:
                    parsed_bounds["set_at"] = bounds_ts.isoformat()
                trigger_bounds = parsed_bounds

        # Priority 2: baseline calibration summary (fallback for older logs).
        elif event_type in ("baseline_calibration_completed", "baseline_calibration_complete"):
            event_ts = _event_timestamp(row)
            if event_ts is not None and baseline_completed_ts is None:
                baseline_completed_ts = event_ts

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
                bounds_ts = _event_timestamp(row)
                if bounds_ts is not None:
                    trigger_bounds["set_at"] = bounds_ts.isoformat()
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
                bounds_ts = _event_timestamp(row)
                if bounds_ts is not None:
                    trigger_bounds["set_at"] = bounds_ts.isoformat()

        # Priority 3: threshold-met event payload (fallback for older logs).
        elif trigger_bounds is None and event_type == "feedback_delivery_threshold_met":
            lower = _to_float(payload.get("lower_bound"))
            upper = _to_float(payload.get("upper_bound"))
            rule = str(payload.get("rule", ""))
            if lower is not None and upper is not None:
                trigger_bounds = {
                    "rule": rule or "baseline_mean_pm_2sd",
                    "lower": lower,
                    "upper": upper,
                }
                bounds_ts = _event_timestamp(row)
                if bounds_ts is not None:
                    trigger_bounds["set_at"] = bounds_ts.isoformat()
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

        if event_type not in ("user_state_estimate_logged", "observer_user_state_estimate_logged"):
            continue

        score = payload.get("score")
        if score is None:
            continue

        ts = _parse_timestamp(payload.get("timestamp"))
        if ts is None:
            ts = _parse_timestamp(row["timestamp"])
        if ts is None or pd.isna(ts):
            continue
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
    interactions_df = pd.DataFrame(feedback_interactions)
    if not interactions_df.empty:
        interactions_df = interactions_df.sort_values("timestamp")

    if trigger_bounds is None:
        # No trigger/baseline events were logged (common in replay); use a neutral default band.
        trigger_bounds = _DEFAULT_TRIGGER_BOUNDS.copy()

    if baseline_completed_ts is not None:
        trigger_bounds["baseline_completed_at"] = baseline_completed_ts.isoformat()

    if tidy.empty:
        return tidy, trigger_bounds, interactions_df

    return tidy.sort_values("timestamp"), trigger_bounds, interactions_df


def _nearest_score_at_timestamp(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[float]:
    if df.empty:
        return None

    if pd.isna(ts):
        return None

    sorted_df = df.sort_values("timestamp")
    timestamp_series = pd.to_datetime(sorted_df["timestamp"], utc=True, errors="coerce")
    score_series = pd.to_numeric(sorted_df["score"], errors="coerce")

    valid_mask = (~timestamp_series.isna()) & (~score_series.isna())
    if not valid_mask.any():
        return None

    ts_utc = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(ts_utc):
        return None

    x = timestamp_series.loc[valid_mask].astype("int64").reset_index(drop=True)
    y = score_series.loc[valid_mask].astype(float).reset_index(drop=True)
    if x.empty:
        return None

    target = int(ts_utc.value)

    # Clamp outside range to nearest endpoint.
    if target <= int(x.iloc[0]):
        return float(y.iloc[0])
    if target >= int(x.iloc[-1]):
        return float(y.iloc[-1])

    right = int(x.searchsorted(target, side="left"))
    left = max(0, right - 1)

    x0 = int(x.iloc[left])
    x1 = int(x.iloc[right])
    y0 = float(y.iloc[left])
    y1 = float(y.iloc[right])

    if x1 == x0:
        return y1

    ratio = (target - x0) / (x1 - x0)
    return y0 + (ratio * (y1 - y0))


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


def plot_estimates(
    df: pd.DataFrame,
    title: str,
    trigger_bounds: Optional[dict[str, float | str]] = None,
    feedback_interactions: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted_reference_rows: list[dict[str, Any]] = []

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
        # smoothed_score = _smooth_series(seg_sorted["score"], method="ema", alpha=0.25)
        smoothed_score = _smooth_series(seg_sorted["score"], method="rolling", alpha=0.25)
        # smoothed_score = seg_sorted["score"]

        ax.plot(
            seg_sorted["timestamp"],
            smoothed_score,
            label=label,
            color=_color_for_mode(mode),
            linewidth=1.8,
        )
        plotted_reference_rows.extend(
            {
                "timestamp": ts,
                "score": score,
            }
            for ts, score in zip(seg_sorted["timestamp"], smoothed_score)
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
        plotted_reference_rows.extend(
            {
                "timestamp": ts,
                "score": score,
            }
            for ts, score in zip(predicted["timestamp"], predicted["score"])
        )

    if trigger_bounds:
        rule = str(trigger_bounds.get("rule", ""))
        lower = _to_float(trigger_bounds.get("lower"))
        upper = _to_float(trigger_bounds.get("upper"))
        threshold = _to_float(trigger_bounds.get("threshold"))
        mean = _to_float(trigger_bounds.get("mean"))
        set_at_ts = _parse_timestamp(trigger_bounds.get("set_at"))
        baseline_completed_line_ts = _parse_timestamp(trigger_bounds.get("baseline_completed_at"))

        if lower is not None and upper is not None:
            band_label = (
                "Trigger band (2.5–97.5 pct)"
                if rule == "baseline_empirical_p2_5_p97_5"
                else "Trigger band (mean ± 2 SD)"
            )
            ax.axhspan(lower, upper, color="#999999", alpha=0.1, zorder=0)
            ax.axhline(lower, color="#b22222", linestyle=":", linewidth=1.2)
            ax.axhline(upper, color="#b22222", linestyle=":", linewidth=1.2)
            if mean is not None:
                ax.axhline(mean, color="#6a6a6a", linestyle="-.", linewidth=1.1)
        elif threshold is not None:
            ax.axhline(
                threshold,
                color="#b22222",
                linestyle=":",
                linewidth=1.2,
            )

        if set_at_ts is not None:
            ax.axvline(
                set_at_ts,
                color="#5a5a5a",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                label="Trigger bounds set",
            )

        if baseline_completed_line_ts is not None:
            ax.axvline(
                baseline_completed_line_ts,
                color="#3f6db3",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                label="Baseline completed",
            )

    if feedback_interactions is not None and not feedback_interactions.empty:
        plotted_reference_df = pd.DataFrame(plotted_reference_rows)
        interaction_legend_added = False
        for _, interaction in feedback_interactions.iterrows():
            marker_ts = interaction["timestamp"]
            marker = str(interaction.get("marker", "")).strip().upper()
            if not marker:
                continue

            marker_y = _nearest_score_at_timestamp(plotted_reference_df, marker_ts)
            if marker_y is None:
                continue

            legend_label = "Feedback interactions" if not interaction_legend_added else None
            interaction_legend_added = True

            ax.scatter(
                [marker_ts],
                [marker_y],
                s=22,
                color="#111111",
                zorder=6,
                label=legend_label,
            )
            ax.annotate(
                marker,
                (marker_ts, marker_y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#111111",
                zorder=7,
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
    parser.add_argument(
        "--feedback-interactions",
        action="store_true",
        help="Overlay logged feedback interactions as single-point markers with one-character labels",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(__file__).resolve().parent / "figures"

    csv_files = _select_files(log_dir, args.pattern, args.session_id, args.file, args.include_all)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[plt.Figure] = []
    for csv_path in csv_files:
        df, trigger_bounds, feedback_interactions = _parse_estimates(csv_path)
        if df.empty:
            continue
        fig = plot_estimates(
            df,
            title=csv_path.stem,
            trigger_bounds=trigger_bounds,
            feedback_interactions=feedback_interactions if args.feedback_interactions else None,
        )
        figures.append(fig)
        output_path = output_dir / f"{csv_path.stem}.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved {output_path}")

    if args.show and figures:
        plt.show()


if __name__ == "__main__":
    main()
