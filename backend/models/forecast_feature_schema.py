"""
Shared feature schema for forecasting training and inference.
"""
import math

# XGBoost input features: exactly five calculated contributor values.
FEATURE_COLUMNS = [
    "contrib_ipa",
    "contrib_fixation_duration",
    "contrib_anticipation",
    "contrib_perceived_difficulty",
    "contrib_ipi",
]

# Forecast targets: predict contributor space directly (same 5 values).
TARGET_COLUMNS = FEATURE_COLUMNS.copy()


def _ramp(value: float, lo: float, hi: float) -> float:
    """Clamp and linearly scale a value into [0, 1]."""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _coerce_float_or_none(value):
    try:
        if value is None:
            return None
        coerced = float(value)
        if math.isnan(coerced):
            return None
        return coerced
    except (TypeError, ValueError):
        return None


def compute_contributor_features(window_features: dict) -> dict:
    """
    Compute the 5 contributor values used as forecasting model inputs.

    Missing inputs fall back to neutral 0.5, matching strict 5-component logic.
    """
    # If contributor values already exist, use them directly.
    # This lets training/evaluation work in contributor space without remapping.
    contrib_ipa_direct = _coerce_float_or_none(window_features.get("contrib_ipa"))
    contrib_fixation_direct = _coerce_float_or_none(window_features.get("contrib_fixation_duration"))
    contrib_anticipation_direct = _coerce_float_or_none(window_features.get("contrib_anticipation"))
    contrib_perceived_direct = _coerce_float_or_none(window_features.get("contrib_perceived_difficulty"))
    contrib_ipi_direct = _coerce_float_or_none(window_features.get("contrib_ipi"))

    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    if (
        contrib_ipa_direct is not None
        and contrib_fixation_direct is not None
        and contrib_anticipation_direct is not None
        and contrib_perceived_direct is not None
        and contrib_ipi_direct is not None
    ):
        return {
            "contrib_ipa": _clamp01(contrib_ipa_direct),
            "contrib_fixation_duration": _clamp01(contrib_fixation_direct),
            "contrib_anticipation": _clamp01(contrib_anticipation_direct),
            "contrib_perceived_difficulty": _clamp01(contrib_perceived_direct),
            "contrib_ipi": _clamp01(contrib_ipi_direct),
        }

    ipa_raw = _coerce_float_or_none(window_features.get("pupil_ipa"))
    fixation_raw = _coerce_float_or_none(window_features.get("fixation_mean_duration_ms"))
    anticipation_raw = _coerce_float_or_none(window_features.get("saccade_mean_velocity"))
    perceived_raw = _coerce_float_or_none(window_features.get("saccade_velocity_std"))
    ipi_raw = _coerce_float_or_none(window_features.get("ipi_value"))

    contrib_ipa = 0.5 if ipa_raw is None else _ramp(ipa_raw, lo=0.5, hi=2.5)
    contrib_fixation = 0.5 if fixation_raw is None else _ramp(fixation_raw, lo=150.0, hi=500.0)
    contrib_anticipation = 0.5 if anticipation_raw is None else _ramp(anticipation_raw, lo=1.0, hi=5.0)
    contrib_perceived = 0.5 if perceived_raw is None else _ramp(perceived_raw, lo=0.5, hi=10.0)
    contrib_ipi = 0.5 if ipi_raw is None else 1.0 - _ramp(ipi_raw, lo=0.5, hi=2.0)

    return {
        "contrib_ipa": contrib_ipa,
        "contrib_fixation_duration": contrib_fixation,
        "contrib_anticipation": contrib_anticipation,
        "contrib_perceived_difficulty": contrib_perceived,
        "contrib_ipi": contrib_ipi,
    }


def compute_score_from_target_components(target_features: dict) -> float:
    """
    Compute strict 5x0.2 score from component values.

    Works for both contributor targets and legacy raw metric targets.
    """
    contribs = compute_contributor_features(target_features)
    return float(
        0.2 * contribs["contrib_ipa"]
        + 0.2 * contribs["contrib_fixation_duration"]
        + 0.2 * contribs["contrib_anticipation"]
        + 0.2 * contribs["contrib_perceived_difficulty"]
        + 0.2 * contribs["contrib_ipi"]
    )
