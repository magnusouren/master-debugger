"""
Type definitions for code context from VS Code.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostics."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"

    @classmethod
    def from_value(cls, value: Any) -> "DiagnosticSeverity":
        """
        Accepts strings like "error"/"warning" or VS Code numeric severities if you ever pass them.
        """
        if isinstance(value, DiagnosticSeverity):
            return value
        if isinstance(value, str):
            v = value.lower()
            try:
                return cls(v)
            except ValueError:
                # fall back
                return cls.INFO
        # Optional: handle VS Code numeric severities (0..3) if that ever happens
        if isinstance(value, int):
            mapping = {
                0: cls.ERROR,
                1: cls.WARNING,
                2: cls.INFO,
                3: cls.HINT,
            }
            return mapping.get(value, cls.INFO)
        return cls.INFO


@dataclass
class CodePosition:
    """A position in a text document."""
    line: int  # 0-indexed line number
    character: int  # 0-indexed character position

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodePosition":
        return cls(
            line=int(data.get("line", 0)),
            character=int(data.get("character", 0)),
        )


@dataclass
class CodeRange:
    """A range in a text document."""
    start: CodePosition
    end: CodePosition
    content: Optional[str] = None  # Optional content of the range

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeRange":
        """Create CodeRange from dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"CodeRange.from_dict expected dict, got {type(data).__name__}")

        return cls(
            start=CodePosition.from_dict(data["start"]),
            end=CodePosition.from_dict(data["end"]),
            content=data.get("content"),
        )


@dataclass
class DiagnosticInfo:
    """A diagnostic message from VS Code (error, warning, etc.)."""
    message: str
    severity: DiagnosticSeverity
    range: CodeRange
    source: Optional[str] = None  # e.g., "typescript", "eslint"
    code: Optional[str] = None  # Error code

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosticInfo":
        if not isinstance(data, dict):
            raise TypeError(f"DiagnosticInfo.from_dict expected dict, got {type(data).__name__}")

        range_data = data.get("range")
        if range_data is None or not isinstance(range_data, dict):
            raise KeyError("DiagnosticInfo missing required field: range")

        return cls(
            message=str(data.get("message", "")),
            severity=DiagnosticSeverity.from_value(data.get("severity", "info")),
            range=CodeRange.from_dict(data["range"]),
            source=data.get("source"),
            code=str(data["code"]) if "code" in data and data["code"] is not None else None,
        )


@dataclass
class CodeContext:
    """
    Complete code context captured from VS Code.
    Sent to the backend for feedback generation.
    """
    # File information
    file_path: str  # Absolute path to the file
    language_id: str  # e.g., "python", "typescript"

    # Cursor and selection
    cursor_position: CodePosition

    # Optional fields (must come after required fields)
    file_content: Optional[str] = None  # Optional full content of the file
    selection: Optional[CodeRange] = None

    # Visible range in editor
    visible_range: Optional[CodeRange] = None

    # Diagnostics
    diagnostics: List[DiagnosticInfo] = field(default_factory=list)

    # Workspace information
    workspace_folder: Optional[str] = None

    # Timestamp when context was captured
    timestamp: float = 0.0

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeContext":
        if not isinstance(data, dict):
            raise TypeError(f"CodeContext.from_dict expected dict, got {type(data).__name__}")

        # Support both new snake_case and older variants you had earlier
        file_path = data.get("file_path") or data.get("fileUri") or data.get("filePath")
        language_id = data.get("language_id") or data.get("languageId")
        cursor_raw = data.get("cursor_position") or data.get("cursor")

        if file_path is None:
            raise KeyError("CodeContext missing required field: file_path")
        if language_id is None:
            raise KeyError("CodeContext missing required field: language_id")
        if cursor_raw is None:
            raise KeyError("CodeContext missing required field: cursor_position")

        selection_raw = data.get("selection")
        visible_raw = data.get("visible_range") or data.get("visibleRange")

        diagnostics_raw = data.get("diagnostics", [])
        if diagnostics_raw is None:
            diagnostics_raw = []

        return cls(
            file_path=str(file_path),
            language_id=str(language_id),
            cursor_position=CodePosition.from_dict(cursor_raw),
            file_content=data.get("file_content") or data.get("fileContent"),
            selection=CodeRange.from_dict(selection_raw) if isinstance(selection_raw, dict) else None,
            visible_range=CodeRange.from_dict(visible_raw) if isinstance(visible_raw, dict) else None,
            diagnostics=[
                DiagnosticInfo.from_dict(d)
                for d in diagnostics_raw
                if isinstance(d, dict)
            ],
            workspace_folder=data.get("workspace_folder") or data.get("workspaceFolder"),
            timestamp=float(data.get("timestamp", 0.0) or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )