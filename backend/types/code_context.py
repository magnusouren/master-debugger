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


@dataclass
class CodePosition:
    """A position in a text document."""
    line: int  # 0-indexed line number
    character: int  # 0-indexed character position


@dataclass
class CodeRange:
    """A range in a text document."""
    start: CodePosition
    end: CodePosition
    content: Optional[str] = None  # Optional content of the range
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeRange":
        """Create CodeRange from dictionary."""
        pass  # TODO: Implement


@dataclass
class DiagnosticInfo:
    """A diagnostic message from VS Code (error, warning, etc.)."""
    message: str
    severity: DiagnosticSeverity
    range: CodeRange
    source: Optional[str] = None  # e.g., "typescript", "eslint"
    code: Optional[str] = None  # Error code


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
