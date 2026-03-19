"""
Logger Service

Provides structured logging with two main categories:
1. Experiment Logging - Research data and event tracking
2. System Logging - Technical/debugging information

Supports configurable log levels and thresholds per category.
"""
import csv
import math
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from backend.api.serialization import json_safe
from backend.types.feedback import FeedbackItem

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
    category: str  # "experiment", "system", or "features"
    mode: Optional[str] = None
    seconds_since_start: Optional[float] = None

@dataclass
class LogFeedbackItem:
    """A log entry for generated feedback."""
    timestamp: float
    event_type: str
    feedback_item: Optional[FeedbackItem] = None
    feedback_id: Optional[str] = None
    seconds_since_start: Optional[float] = None

class LoggerService:
    """
    Centralized logging service for experiment and system logs.
    """
    
    def __init__(
        self,
        experiment_level: str = "INFO",
        system_level: str = "INFO",
        max_entries: int = 100000,
        experiment_mode: str = None,
        features_level: str = "INFO",
    ):
        """
        Initialize the logger service.
        
        Args:
            experiment_level: Log level for experiment logs (DEBUG, INFO, WARNING, ERROR).
            system_level: Log level for system logs (DEBUG, INFO, WARNING, ERROR).
            max_entries: Maximum entries per category before rotating.
            experiment_mode: Operation mode for experiment logs (e.g., "reactive", "proactive").
        """
        self.experiment_logs: List[LogEntry] = []
        self.system_logs: List[LogEntry] = []
        self.feature_logs: List[LogEntry] = []
        self.feedback_logs: List[LogFeedbackItem] = []
        
        self.experiment_level = LogLevel[experiment_level.upper()]
        self.system_level = LogLevel[system_level.upper()]
        self.features_level = LogLevel[features_level.upper()]
        
        self.max_entries = max_entries
        self.experiment_mode = experiment_mode
        self.start_time: Optional[float] = None

    def set_experiment_mode(self, mode: str) -> None:
        """
        Set the operation mode, used for correct logging.
        
        """
        self.experiment_mode = mode

    def set_start_time(self, start_time: Optional[float] = None) -> None:
        """
        Set or reset the reference start time for seconds-since-start logging.
        If start_time is None, uses current UTC time.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc).timestamp()
        self.start_time = float(start_time)

    def reset(self) -> None:
        """Clear all in-memory logs so a new experiment starts fresh."""
        self.experiment_logs.clear()
        self.system_logs.clear()
        self.feature_logs.clear()
        self.feedback_logs.clear()
        self.start_time = None
        self.experiment_mode = None

    def _seconds_since_start(self, ts: float) -> Optional[float]:
        """Compute seconds since start_time, if set."""
        if self.start_time is None:
            return None
        return ts - self.start_time
    
    def experiment(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        """
        Log an experiment event.
        
        """
        if not self._should_log(level, category="experiment"):
            return

        ts = datetime.now(timezone.utc).timestamp()
        seconds_since_start = self._seconds_since_start(ts)

        entry = LogEntry(
            timestamp=ts,
            level=level.upper(),
            event_type=event_type,
            data=data or {},
            category="experiment",
            mode=self.experiment_mode,
            seconds_since_start=seconds_since_start,
        )
        
        self.experiment_logs.append(entry)
        
        # Rotate if exceeding max entries
        if len(self.experiment_logs) > self.max_entries:
            self.experiment_logs = self.experiment_logs[-self.max_entries:]

    def features(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        """
        Log a feature stream event (high-volume window data).
        """
        if not self._should_log(level, category="features"):
            return

        ts = datetime.now(timezone.utc).timestamp()
        seconds_since_start = self._seconds_since_start(ts)

        entry = LogEntry(
            timestamp=ts,
            level=level.upper(),
            event_type=event_type,
            data=data or {},
            category="features",
            mode=self.experiment_mode,
            seconds_since_start=seconds_since_start,
        )

        self.feature_logs.append(entry)

        if len(self.feature_logs) > self.max_entries:
            self.feature_logs = self.feature_logs[-self.max_entries:]
    
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

        ts = datetime.now(timezone.utc).timestamp()
        seconds_since_start = self._seconds_since_start(ts)

        entry = LogEntry(
            timestamp=ts,
            level=level.upper(),
            event_type=event_type,
            data=data or {},
            category="system",
            mode=self.experiment_mode,
            seconds_since_start=seconds_since_start,
        )
        
        self.system_logs.append(entry)
        self._print_log(entry)
        
        # Rotate if exceeding max entries
        if len(self.system_logs) > self.max_entries:
            self.system_logs = self.system_logs[-self.max_entries:]

    def feedback(
        self,
        event_type: str,
        feedback_item: FeedbackItem,
    ) -> None:
        """
        Log a generated feedback item.
        
        Args:
            feedback_item: The FeedbackItem instance to log.
        """
        ts = datetime.now(timezone.utc).timestamp()
        seconds_since_start = self._seconds_since_start(ts)

        entry = LogFeedbackItem(
            timestamp=ts,
            event_type=event_type,
            feedback_id=feedback_item.metadata.feedback_id if feedback_item.metadata else "unknown",
            feedback_item=feedback_item,
            seconds_since_start=seconds_since_start,
        )
        self.feedback_logs.append(entry)

        # Rotate if exceeding max entries        
        if len(self.feedback_logs) > self.max_entries:
            self.feedback_logs = self.feedback_logs[-self.max_entries:]
    
    def set_level(self, category: str, level: str) -> None:
        """
        Set log level threshold for a category.
        
        Args:
            category: "experiment", "system", or "features".
            level: "DEBUG", "INFO", "WARNING", or "ERROR".
        """
        level_obj = LogLevel[level.upper()]
        
        if category.lower() == "experiment":
            self.experiment_level = level_obj
        elif category.lower() == "system":
            self.system_level = level_obj
        elif category.lower() == "features":
            self.features_level = level_obj
        else:
            raise ValueError(f"Unknown category: {category}")
    
    def export_experiment_logs(self, filepath: str) -> bool:
        """
        Export experiment logs to CSV file.
        
        Args:
            filepath: Path to export file.
            
        Returns:
            True if successful.

        """
        try:
            filepath = filepath if filepath.endswith(".csv") else f"{filepath}.csv"
            filepath = Path(filepath)

            # Ensure parent directories exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            logs_data = [
                {
                    "timestamp": entry.timestamp,
                    "seconds_since_start": entry.seconds_since_start,
                    "level": entry.level,
                    "mode": entry.mode,
                    "event_type": entry.event_type,
                    "data": entry.data,
                }
                for entry in self.experiment_logs
            ]
            
            
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "seconds_since_start", "level", "mode", "event_type", "data"])

                for entry in logs_data:
                    dt = datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc)
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    writer.writerow([
                        timestamp_str,
                        round(entry["seconds_since_start"], 3) if entry["seconds_since_start"] is not None else "",
                        entry["level"],
                        entry["mode"].upper() if entry["mode"] else "UNKNOWN",
                        entry["event_type"],
                        json.dumps(json_safe(entry["data"])),
                    ])

            self._print_log(
                LogEntry(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    level="INFO",
                    event_type="export_experiment_logs",
                    mode=self.experiment_mode,
                    data={"filepath": filepath, "count": len(logs_data)},
                    category="system",
                )
            )
            return True
        except Exception as e:
            self._print_log(
                LogEntry(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    level="ERROR",
                    mode=self.experiment_mode,
                    event_type="export_experiment_logs_error",
                    data={"error": str(e)},
                    category="system",
                )
            )
            return False
        
    
    def export_system_logs(self, filepath: str) -> bool:
        """
        Export system logs to CSV file.
        
        Args:
            filepath: Path to export file.
            
        Returns:
            True if successful.
        """
        try:
            filepath = filepath if filepath.endswith(".csv") else f"{filepath}.csv"
            filepath = Path(filepath)

            # Ensure parent directories exist
            filepath.parent.mkdir(parents=True, exist_ok=True)


            logs_data = [
                {
                    "timestamp": entry.timestamp,
                    "seconds_since_start": entry.seconds_since_start,
                    "level": entry.level,
                    "mode": entry.mode,
                    "event_type": entry.event_type,
                    "data": entry.data,
                }
                for entry in self.system_logs
            ]

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "seconds_since_start", "level", "mode", "event_type", "data"])

                for entry in logs_data:
                    dt = datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc)
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    writer.writerow([
                        timestamp_str,
                        round(entry["seconds_since_start"], 3) if entry["seconds_since_start"] is not None else "",
                        entry["level"],
                        entry["mode"].upper() if entry["mode"] else "UNKNOWN",
                        entry["event_type"],
                        json.dumps(json_safe(entry["data"])),
                    ])
            
            self._print_log(
                LogEntry(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    level="INFO",
                    mode=self.experiment_mode,
                    event_type="export_system_logs",
                    data={"filepath": filepath, "count": len(logs_data)},
                    category="system",
                )
            )
            return True
        except Exception as e:
            self._print_log(
                LogEntry(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    level="ERROR",
                    mode=self.experiment_mode,
                    event_type="export_system_logs_error",
                    data={"error": str(e)},
                    category="system",
                )
            )
            return False
        
    def export_feedback_logs(self, filepath: str) -> bool:
        """
        Export feedback logs to CSV file.
        
        Args:
            filepath: Path to export file.
            
        Returns:
            True if successful.
        """
        try:
            filepath = filepath if filepath.endswith(".csv") else f"{filepath}.csv"
            filepath = Path(filepath)

            # Ensure parent directories exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            logs_data = [
                {
                    "timestamp": entry.timestamp,
                    "seconds_since_start": entry.seconds_since_start,
                    "event_type": entry.event_type,
                    "feedback_id": entry.feedback_id,
                    "feedback_item": entry.feedback_item,
                }
                for entry in self.feedback_logs
            ]

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "seconds_since_start", "event_type", "feedback_id", "feedback_item"])

                for entry in logs_data:
                    dt = datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc)
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    writer.writerow([
                        timestamp_str,
                        round(entry["seconds_since_start"], 3) if entry["seconds_since_start"] is not None else "",
                        entry["event_type"],
                        entry["feedback_id"],
                        json.dumps(json_safe(entry["feedback_item"] or {})),
                    ])
            
            self._print_log(
                LogEntry(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    level="INFO",
                    mode=self.experiment_mode,
                    event_type="export_feedback_logs",
                    data={"filepath": filepath, "count": len(logs_data)},
                    category="system",
                )
            )
            return True
        except Exception as e:
            self._print_log(
                LogEntry(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    level="ERROR",
                    mode=self.experiment_mode,
                    event_type="export_feedback_logs_error",
                    data={"error": str(e)},
                    category="system",
                )
            )
            return False

    def export_feature_logs(self, filepath: str) -> bool:
        """
        Export feature stream logs to CSV file.

        Args:
            filepath: Path to export file.

        Returns:
            True if successful.
        """
        try:
            filepath = filepath if filepath.endswith(".csv") else f"{filepath}.csv"
            filepath = Path(filepath)

            filepath.parent.mkdir(parents=True, exist_ok=True)

            logs_data = [
                {
                    "timestamp": entry.timestamp,
                    "seconds_since_start": entry.seconds_since_start,
                    "level": entry.level,
                    "mode": entry.mode,
                    "event_type": entry.event_type,
                    "data": entry.data,
                }
                for entry in self.feature_logs
            ]

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "seconds_since_start", "level", "mode", "event_type", "data"])

                for entry in logs_data:
                    dt = datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc)
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    writer.writerow([
                        timestamp_str,
                        round(entry["seconds_since_start"], 3) if entry["seconds_since_start"] is not None else "",
                        entry["level"],
                        entry["mode"].upper() if entry["mode"] else "UNKNOWN",
                        entry["event_type"],
                        json.dumps(json_safe(entry["data"])),
                    ])

            self._print_log(
                LogEntry(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    level="INFO",
                    event_type="export_feature_logs",
                    mode=self.experiment_mode,
                    data={"filepath": filepath, "count": len(logs_data)},
                    category="system",
                )
            )
            return True
        except Exception as e:
            self._print_log(
                LogEntry(
                    timestamp=datetime.now(timezone.utc).timestamp(),
                    level="ERROR",
                    mode=self.experiment_mode,
                    event_type="export_feature_logs_error",
                    data={"error": str(e)},
                    category="system",
                )
            )
            return False
    
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
        elif category.lower() == "features":
            threshold = self.features_level
        else:
            # Default to experiment threshold for "experiment" and any other categories.
            threshold = self.experiment_level
        
        return level_obj.value >= threshold.value
    
    def _print_log(self, entry: LogEntry) -> None:
        timestamp = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
        delta_str = ""
        if entry.seconds_since_start is not None:
            delta_str = f" +{entry.seconds_since_start:.3f}s"

        colors = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
        }
        reset = "\033[0m"
        color = colors.get(entry.level, "")

        mode = entry.mode.upper() if entry.mode else "UNKNOWN"

        data_obj = json_safe(entry.data) if entry.data else None
        data_str = json.dumps(data_obj) if data_obj else ""

        line = f"{color}[{timestamp}{delta_str}] [{entry.level}] [{mode}] {entry.event_type}{reset} {data_str}"
        
        print(line, flush=True)


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
    features_level: str = "INFO",
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
        features_level=features_level,
    )
    return _logger
