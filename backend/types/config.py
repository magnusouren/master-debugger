"""
Configuration type definitions for all system layers.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, List, Dict, Any, Type, TypeVar, get_origin, get_args, Union
from enum import Enum
import yaml

T = TypeVar("T")

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _is_optional(tp: Any) -> bool:
    origin = get_origin(tp)
    return origin is Union and type(None) in get_args(tp)


def _strip_optional(tp: Any) -> Any:
    return next(t for t in get_args(tp) if t is not type(None))


def _coerce_value(value: Any, target_type: Any) -> Any:
    """Convert YAML value into the target field type."""
    if value is None:
        return None

    # Optional[T]
    if _is_optional(target_type):
        return _coerce_value(value, _strip_optional(target_type))

    # Enum
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(value)

    # Nested dataclass
    if isinstance(target_type, type) and is_dataclass(target_type):
        return _dict_to_dataclass(value, target_type)

    origin = get_origin(target_type)

    # List[T]
    if origin in (list, List):
        (item_type,) = get_args(target_type)
        return [_coerce_value(v, item_type) for v in value]

    # Dict[K, V]
    if origin in (dict, Dict):
        key_type, val_type = get_args(target_type)
        return {
            _coerce_value(k, key_type): _coerce_value(v, val_type)
            for k, v in value.items()
        }

    return value


def _dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """Create dataclass instance from dict (ignores unknown keys)."""
    field_map = {f.name: f for f in fields(cls)}
    kwargs = {}

    for key, value in data.items():
        if key not in field_map:
            continue  # ignore unknown config keys
        field_type = field_map[key].type
        kwargs[key] = _coerce_value(value, field_type)

    return cls(**kwargs)


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass to YAML-safe dict."""
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj

#------------------------------------------------------------------
# Configuration Data Classes
#------------------------------------------------------------------


class OperationMode(Enum):
    """System operation modes."""
    REACTIVE = "reactive"  # Respond to current state
    PROACTIVE = "proactive"  # Predict and preempt


@dataclass
class SignalProcessingConfig:
    """Configuration for Signal Processing layer."""
    # Input settings
    input_sampling_rate_hz: float = 60.0
    
    # Window settings
    window_length_seconds: float = 1.0
    window_overlap_ratio: float = 0.5
    
    # Output settings
    output_frequency_hz: float = 5.0  # 2-10 Hz as per requirements
    
    # Metrics to extract (TODO: define specific metrics)
    enabled_metrics: List[str] = field(default_factory=lambda: [
        "fixation_duration",
        "saccade_amplitude",
        "pupil_diameter",
        "blink_rate",
        "gaze_dispersion",
    ])
    
    # Data quality settings
    min_valid_sample_ratio: float = 0.5
    interpolate_missing: bool = True
    max_gap_to_interpolate_ms: float = 100.0


@dataclass
class ForecastingConfig:
    """Configuration for Forecasting Tool (proactive mode)."""
    enabled: bool = False  # Only active in proactive mode
    
    # Prediction settings
    prediction_horizon_seconds: float = 2.0
    update_rate_hz: float = 5.0
    
    # Model settings
    model_path: Optional[str] = None
    model_type: str = "lstm"  # or "transformer", "arima", etc.
    
    # History window for predictions
    history_window_seconds: float = 5.0
    
    # Confidence threshold
    min_confidence_threshold: float = 0.5


@dataclass
class ReactiveToolConfig:
    """Configuration for Reactive Tool."""
    # Sliding window settings
    window_size_seconds: float = 3.0
    
    # Model settings
    model_type: str = "rule_based"  # "rule_based", "ml_classifier", "sequence_model"
    model_path: Optional[str] = None
    
    # Thresholds for rule-based approach
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high_load": 0.7,
        "medium_load": 0.4,
        "low_load": 0.2,
    })
    
    # Output settings
    score_smoothing_factor: float = 0.3  # EMA smoothing
    min_confidence_for_action: float = 0.6


@dataclass
class FeedbackLayerConfig:
    """Configuration for Feedback Layer."""
    # LLM settings
    llm_provider: Optional[str] = None  # "openai", "anthropic", "local"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    
    # Generation settings
    max_feedback_items: int = 3
    max_message_length: int = 200
    
    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: float = 300.0
    
    # Rate limiting
    max_generations_per_minute: int = 10


@dataclass
class ControllerConfig:
    """Configuration for Runtime Controller."""
    # Mode
    operation_mode: OperationMode = OperationMode.REACTIVE
    
    # Feedback timing
    feedback_cooldown_seconds: float = 5.0
    min_score_for_feedback: float = 0.6
    
    # WebSocket settings
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    
    # API settings
    api_host: str = "localhost"
    api_port: int = 8080
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: Optional[str] = None
    
    # Experiment settings
    experiment_id: Optional[str] = None
    participant_id: Optional[str] = None


@dataclass
class SystemConfig:
    """Complete system configuration."""
    signal_processing: SignalProcessingConfig = field(default_factory=SignalProcessingConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    reactive_tool: ReactiveToolConfig = field(default_factory=ReactiveToolConfig)
    feedback_layer: FeedbackLayerConfig = field(default_factory=FeedbackLayerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    @classmethod
    def from_file(cls, path: str) -> "SystemConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping")

        return _dict_to_dataclass(data, cls)

    def to_file(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = _dataclass_to_dict(self)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                default_flow_style=False,
            )