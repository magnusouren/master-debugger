from dataclasses import is_dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any

def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return {k: json_safe(v) for k, v in asdict(x).items()}
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x