"""
Logger Service

Provides structured logging with two main categories:
1. Experiment Logging - Research data and event tracking
2. System Logging - Technical/debugging information

Supports configurable log levels and thresholds per category.
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
import json


class LogLevel(Enum):
    """Log level hierarchy (ascending verbosity)."""
    ERROR = 4
    WARNING = 3
    INFO = 2
    DEBUG = 1


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: float
    level: str
    event_type: str
    data: Dict[str, Any]
    category: str  # "experiment" or "system"


class LoggerService:
    """
    Centralized logging service for experiment and system logs.
    """
    
    def __init__(
        self,
        experiment_level: str = "INFO",
        system_level: str = "INFO",
        max_entries: int = 10000,
    ):
        """
        Initialize the logger service.
        
        Args:
            experiment_level: Log level for experiment logs (DEBUG, INFO, WARNING, ERROR).
            system_level: Log level for system logs (DEBUG, INFO, WARNING, ERROR).
            max_entries: Maximum entries per category before rotating.
        """
        self.experiment_logs: List[LogEntry] = []
        self.system_logs: List[LogEntry] = []
        
        self.experiment_level = LogLevel[experiment_level.upper()]
        self.system_level = LogLevel[system_level.upper()]
        
        self.max_entries = max_entries
    
    def experiment(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        """
        Log an experiment event.
        
        Args:
            event_type: Type of event (e.g., "feedback_generated", "context_updated").
            data: Event data as dictionary.
            level: Log level (DEBUG, INFO, WARNING, ERROR).
        """
        if not self._should_log(level, category="experiment"):
            return
        
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).timestamp(),
            level=level.upper(),
            event_type=event_type,
            data=data or {},
            category="experiment",
        )
        
        self.experiment_logs.append(entry)
        
        # Rotate if exceeding max entries
        if len(self.experiment_logs) > self.max_entries:
            self.experiment_logs = self.experiment_logs[-self.max_entries:]
    
    def system(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        """
        Log a system event.
        
        Args:
            event_type: Type of event (e.g., "server_started", "connection_error").
            data: Event data as dictionary.
            level: Log level (DEBUG, INFO, WARNING, ERROR).
        """
        if not self._should_log(level, category="system"):
            return
        
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).timestamp(),
            level=level.upper(),
            event_type=event_type,
            data=data or {},
            category="system",
        )
        
        self.system_logs.append(entry)
        self._print_log(entry)
        
        # Rotate if exceeding max entries
        if len(self.system_logs) > self.max_entries:
            self.system_logs = self.system_logs[-self.max_entries:]
    
    def set_level(self, category: str, level: str) -> None:
        """
        Set log level threshold for a category.
        
        Args:
            category: "experiment" or "system".
            level: "DEBUG", "INFO", "WARNING", or "ERROR".
        """
        level_obj = LogLevel[level.upper()]
        
        if category.lower() == "experiment":
            self.experiment_level = level_obj
        elif category.lower() == "system":
            self.system_level = level_obj
        else:
            raise ValueError(f"Unknown category: {category}")
    
    def get_experiment_logs(
        self,
        event_type: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[LogEntry]:
        """
        Retrieve experiment logs with optional filtering.
        
        Args:
            event_type: Filter by event type.
            level: Filter by log level.
            
        Returns:
            List of matching log entries.
        """
        logs = self.experiment_logs
        
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
        
        if level:
            logs = [l for l in logs if l.level == level.upper()]
        
        return logs
    
    def get_system_logs(
        self,
        event_type: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[LogEntry]:
        """
        Retrieve system logs with optional filtering.
        
        Args:
            event_type: Filter by event type.
            level: Filter by log level.
            
        Returns:
            List of matching log entries.
        """
        logs = self.system_logs
        
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
        
        if level:
            logs = [l for l in logs if l.level == level.upper()]
        
        return logs
    
    def clear_logs(self, category: str = "all") -> None:
        """
        Clear logs.
        
        Args:
            category: "experiment", "system", or "all".
        """
        if category.lower() in ("experiment", "all"):
            self.experiment_logs = []
        
        if category.lower() in ("system", "all"):
            self.system_logs = []
    
    def export_experiment_logs(self, filepath: str) -> bool:
        """
        Export experiment logs to JSON file.
        
        Args:
            filepath: Path to export file.
            
        Returns:
            True if successful.
        """
        try:
            logs_data = [
                {
                    "timestamp": entry.timestamp,
                    "level": entry.level,
                    "event_type": entry.event_type,
                    "data": entry.data,
                }
                for entry in self.experiment_logs
            ]
            
            with open(filepath, "w") as f:
                json.dump(logs_data, f, indent=2, default=str)
            
            print(f"[Logger] Exported {len(logs_data)} experiment logs to {filepath}")
            return True
        except Exception as e:
            print(f"[Logger] Error exporting experiment logs: {e}")
            return False
    
    def export_system_logs(self, filepath: str) -> bool:
        """
        Export system logs to JSON file.
        
        Args:
            filepath: Path to export file.
            
        Returns:
            True if successful.
        """
        try:
            logs_data = [
                {
                    "timestamp": entry.timestamp,
                    "level": entry.level,
                    "event_type": entry.event_type,
                    "data": entry.data,
                }
                for entry in self.system_logs
            ]
            
            with open(filepath, "w") as f:
                json.dump(logs_data, f, indent=2, default=str)
            
            print(f"[Logger] Exported {len(logs_data)} system logs to {filepath}")
            return True
        except Exception as e:
            print(f"[Logger] Error exporting system logs: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logging statistics.
        
        Returns:
            Dictionary with log counts and levels.
        """
        exp_by_level = {}
        sys_by_level = {}
        
        for entry in self.experiment_logs:
            exp_by_level[entry.level] = exp_by_level.get(entry.level, 0) + 1
        
        for entry in self.system_logs:
            sys_by_level[entry.level] = sys_by_level.get(entry.level, 0) + 1
        
        return {
            "experiment": {
                "total": len(self.experiment_logs),
                "by_level": exp_by_level,
                "level_threshold": self.experiment_level.name,
            },
            "system": {
                "total": len(self.system_logs),
                "by_level": sys_by_level,
                "level_threshold": self.system_level.name,
            },
        }
    
    # --- Internal Methods ---
    
    def _should_log(self, level: str, category: str = "experiment") -> bool:
        """
        Determine if a message should be logged based on level.
        
        Args:
            level: Message level.
            category: Log category, e.g. "experiment" or "system".
            
        Returns:
            True if message should be logged.
        """
        try:
            level_obj = LogLevel[level.upper()]
        except KeyError:
            # Preserve existing behavior: log unknown levels by default.
            return True  # Log unknown levels
        
        # Choose appropriate threshold based on category.
        if category.lower() == "system":
            threshold = self.system_level
        else:
            # Default to experiment threshold for "experiment" and any other categories.
            threshold = self.experiment_level
        
        return level_obj.value >= threshold.value
    
    def _print_log(self, entry: LogEntry) -> None:
        """
        Pretty print a system log entry to console.
        
        Args:
            entry: Log entry to print.
        """
        timestamp = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
        
        # Color codes for terminal
        colors = {
            "DEBUG": "\033[36m",      # Cyan
            "INFO": "\033[32m",       # Green
            "WARNING": "\033[33m",    # Yellow
            "ERROR": "\033[31m",      # Red
        }
        reset = "\033[0m"
        
        color = colors.get(entry.level, "")
        data_str = json.dumps(entry.data) if entry.data else ""
        
        print(f"{color}[{timestamp}] [{entry.level}] {entry.event_type}{reset} {data_str}")


# Global logger instance
_logger: Optional[LoggerService] = None


def get_logger() -> LoggerService:
    """
    Get the global logger instance.
    
    Returns:
        Global LoggerService instance.
    """
    global _logger
    if _logger is None:
        _logger = LoggerService()
    return _logger


def initialize_logger(
    experiment_level: str = "INFO",
    system_level: str = "INFO",
) -> LoggerService:
    """
    Initialize the global logger service.
    
    Args:
        experiment_level: Log level for experiment logs.
        system_level: Log level for system logs.
        
    Returns:
        Initialized LoggerService instance.
    """
    global _logger
    _logger = LoggerService(
        experiment_level=experiment_level,
        system_level=system_level,
    )
    return _logger
