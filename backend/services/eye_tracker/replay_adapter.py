"""
Replay Eye Tracker Adapter

Reads gaze samples from a TSV export file and replays them as if they were
coming from a live eye tracker.

This version:
- streams at a fixed target frequency (default 250 Hz)
- uses system time only for emitted sample timestamps
- ignores file timing for replay pacing
"""
import asyncio
import csv
import io
import time
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

from backend.services.eye_tracker.base import EyeTrackerAdapter, AdapterState
from backend.types.eye_tracking import GazeSample
from backend.services.logger_service import get_logger


class ReplayEyeTrackerAdapter(EyeTrackerAdapter):
    def __init__(
        self,
        file_path: str,
        batch_size: int = 12,
        flush_interval_ms: int = 16,
        target_hz: float = 250.0,
        timestamp_unit: str = "microseconds",  # only used for parsing file timestamps into raw_data
        fast_forward: bool = False,  # when True: no pacing sleeps, just emit as fast as possible
        use_system_timestamps: bool = True,  # when False: keep file timestamps on emitted samples
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._file_path = Path(file_path)
        self._batch_size = batch_size
        self._flush_interval_ms = flush_interval_ms
        self._target_hz = target_hz
        self._target_dt = 1.0 / target_hz if target_hz > 0 else 0.0
        self._timestamp_unit = timestamp_unit
        self._fast_forward = fast_forward
        self._use_system_timestamps = use_system_timestamps
        self._loop = loop or asyncio.get_event_loop()

        self._state = AdapterState.DISCONNECTED
        self._device_id = f"replay:{self._file_path.name}"

        self._samples_callback: Optional[Callable[[List[GazeSample]], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None

        self._streaming_task: Optional[asyncio.Task] = None
        self._buffer: List[GazeSample] = []
        self._samples: List[GazeSample] = []

        self._logger = get_logger()

    async def connect(self, device_id: Optional[str] = None) -> bool:
        if self._state != AdapterState.DISCONNECTED:
            return False

        self._state = AdapterState.CONNECTING

        try:
            if device_id:
                self._device_id = device_id

            self._samples = self._load_samples_from_file(self._file_path)

            if not self._samples:
                raise ValueError(f"No gaze samples found in file: {self._file_path}")

            self._state = AdapterState.CONNECTED
            return True

        except Exception as e:
            self._state = AdapterState.ERROR
            if self._error_callback:
                self._error_callback(e)
            return False

    async def disconnect(self) -> None:
        if self._state == AdapterState.STREAMING:
            await self.stop_streaming()

        self._samples.clear()
        self._buffer.clear()
        self._state = AdapterState.DISCONNECTED

    def is_connected(self) -> bool:
        return self._state in (AdapterState.CONNECTED, AdapterState.STREAMING)

    def get_device_info(self) -> Dict[str, Any]:
        if not self.is_connected():
            return {}

        return {
            "device_id": self._device_id,
            "model": "Replay Eye Tracker Adapter",
            "address": str(self._file_path),
            "sample_count": len(self._samples),
            "target_hz": self._target_hz,
            "fast_forward": self._fast_forward,
            "use_system_timestamps": self._use_system_timestamps,
            "timestamp_source": "system_time",
            "timestamp_unit_in_file": self._timestamp_unit,
        }

    def set_samples_callback(self, callback: Callable[[List[GazeSample]], None]) -> None:
        self._samples_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        self._error_callback = callback

    async def start_streaming(self) -> None:
        if self._state != AdapterState.CONNECTED:
            raise RuntimeError(
                f"Cannot start streaming: adapter is in {self._state.value} state"
            )

        self._state = AdapterState.STREAMING
        self._buffer.clear()
        self._streaming_task = asyncio.create_task(self._stream_samples())

    async def stop_streaming(self) -> None:
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None

        if self._buffer and self._samples_callback:
            self._samples_callback(self._buffer.copy())

        self._buffer.clear()

        if self._state == AdapterState.STREAMING:
            self._state = AdapterState.CONNECTED

    def get_state(self) -> AdapterState:
        return self._state

    async def _stream_samples(self) -> None:
        flush_interval = self._flush_interval_ms / 1000.0
        last_flush_time = time.time()

        try:
            if not self._fast_forward and self._target_dt <= 0:
                raise ValueError("target_hz must be > 0")

            next_emit_time = time.perf_counter()
            synthetic_ts = time.time()

            for sample in self._samples:
                if self._state != AdapterState.STREAMING:
                    break

                if not self._fast_forward:
                    # Fixed-rate pacing using monotonic time to reduce drift.
                    next_emit_time += self._target_dt
                    sleep_time = next_emit_time - time.perf_counter()
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                # Emit sample with chosen timestamping strategy.
                if self._fast_forward and self._use_system_timestamps:
                    emitted_sample = self._clone_sample_with_timestamp(sample, synthetic_ts)
                    if self._target_dt > 0:
                        synthetic_ts += self._target_dt
                else:
                    emitted_sample = self._prepare_emitted_sample(sample)

                self._buffer.append(emitted_sample)
                current_time = time.time()

                should_flush = (
                    len(self._buffer) >= self._batch_size
                    or (current_time - last_flush_time) >= flush_interval
                )

                if should_flush and self._samples_callback:
                    batch = self._buffer.copy()
                    self._buffer.clear()
                    self._samples_callback(batch)
                    last_flush_time = current_time

                    if self._fast_forward:
                        await asyncio.sleep(0)

            if self._buffer and self._samples_callback:
                self._samples_callback(self._buffer.copy())
                self._buffer.clear()

            self._state = AdapterState.CONNECTED

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._state = AdapterState.ERROR
            if self._error_callback:
                self._error_callback(e)
            raise

    def _clone_sample_with_system_timestamp(self, sample: GazeSample) -> GazeSample:
        raw_data = dict(sample.raw_data) if sample.raw_data else {}
        raw_data["stream_timestamp_source"] = "system_time"
        raw_data["streamed_at"] = time.time()

        return GazeSample(
            timestamp=time.time(),
            left_eye_x=sample.left_eye_x,
            left_eye_y=sample.left_eye_y,
            right_eye_x=sample.right_eye_x,
            right_eye_y=sample.right_eye_y,
            left_pupil_diameter=sample.left_pupil_diameter,
            right_pupil_diameter=sample.right_pupil_diameter,
            left_eye_valid=sample.left_eye_valid,
            right_eye_valid=sample.right_eye_valid,
            raw_data=raw_data,
        )

    def _clone_sample_with_timestamp(self, sample: GazeSample, timestamp: float) -> GazeSample:
        raw_data = dict(sample.raw_data) if sample.raw_data else {}
        raw_data["stream_timestamp_source"] = "synthetic_time"
        raw_data["streamed_at"] = time.time()

        return GazeSample(
            timestamp=timestamp,
            left_eye_x=sample.left_eye_x,
            left_eye_y=sample.left_eye_y,
            right_eye_x=sample.right_eye_x,
            right_eye_y=sample.right_eye_y,
            left_pupil_diameter=sample.left_pupil_diameter,
            right_pupil_diameter=sample.right_pupil_diameter,
            left_eye_valid=sample.left_eye_valid,
            right_eye_valid=sample.right_eye_valid,
            raw_data=raw_data,
        )

    def _prepare_emitted_sample(self, sample: GazeSample) -> GazeSample:
        """Choose emitted timestamp strategy based on config."""
        if self._use_system_timestamps:
            return self._clone_sample_with_system_timestamp(sample)
        return sample

    def _load_samples_from_file(self, file_path: Path) -> List[GazeSample]:
        samples: List[GazeSample] = []

        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            lines = f.readlines()

        header_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Time\tType\tTrial"):
                header_index = i
                break

        if header_index is None:
            raise ValueError("Could not find TSV header line starting with 'Time\\tType\\tTrial'")

        data_text = "".join(lines[header_index:])
        reader = csv.DictReader(io.StringIO(data_text), delimiter="\t")

        for row in reader:
            try:
                if not any((value or "").strip() for value in row.values()):
                    continue

                row_type = (row.get("Type") or "").strip()

                # Keep only actual sample rows
                if row_type != "SMP":
                    continue

                sample = self._row_to_gaze_sample(row)
                samples.append(sample)

            except Exception as e:
                self._logger.system(
                    "replay_adapter_row_parse_failed",
                    {
                        "error": str(e),
                        "row_preview": dict(list(row.items())[:8]),
                    },
                    level="WARNING",
                )

        return samples

    def _row_to_gaze_sample(self, row: Dict[str, str]) -> GazeSample:
        def clean(value: Optional[str]) -> str:
            if value is None:
                return ""
            return value.replace("−", "-").strip()

        def to_float(value: Optional[str], default: float = 0.0) -> float:
            text = clean(value)
            if text == "":
                return default
            return float(text)

        def to_int(value: Optional[str], default: int = 0) -> int:
            text = clean(value)
            if text == "":
                return default
            return int(float(text))

        raw_timestamp = to_float(row.get("Time"))

        if self._timestamp_unit == "seconds":
            file_timestamp_seconds = raw_timestamp
        elif self._timestamp_unit == "milliseconds":
            file_timestamp_seconds = raw_timestamp / 1000.0
        elif self._timestamp_unit == "microseconds":
            file_timestamp_seconds = raw_timestamp / 1_000_000.0
        else:
            raise ValueError(f"Unsupported timestamp unit: {self._timestamp_unit}")

        left_validity = to_int(row.get("L Validity"), default=4)
        right_validity = to_int(row.get("R Validity"), default=4)

        return GazeSample(
            # Initial timestamp from file is only a placeholder before streaming.
            # It will be replaced with system time when emitted.
            timestamp=file_timestamp_seconds,
            left_eye_x=to_float(row.get("L Raw X [px]")),
            left_eye_y=to_float(row.get("L Raw Y [px]")),
            right_eye_x=to_float(row.get("R Raw X [px]")),
            right_eye_y=to_float(row.get("R Raw Y [px]")),
            left_pupil_diameter=to_float(row.get("L Mapped Diameter [mm]")),
            right_pupil_diameter=to_float(row.get("R Mapped Diameter [mm]")),
            # Treat 0 or 1 as valid; anything >1 is invalid.
            left_eye_valid=(left_validity <= 1),
            right_eye_valid=(right_validity <= 1),
            raw_data={
                "type": clean(row.get("Type")),
                "trial": to_int(row.get("Trial"), default=0),
                "timing": to_int(row.get("Timing"), default=0),
                "pupil_confidence": to_int(row.get("Pupil Confidence"), default=0),
                "left_por_x": to_float(row.get("L POR X [px]")),
                "left_por_y": to_float(row.get("L POR Y [px]")),
                "right_por_x": to_float(row.get("R POR X [px]")),
                "right_por_y": to_float(row.get("R POR Y [px]")),
                "frame": clean(row.get("Frame")),
                "aux1": clean(row.get("Aux1")),
                "file_timestamp_raw": raw_timestamp,
                "file_timestamp_seconds": file_timestamp_seconds,
                "source": "replay_file",
            },
        )
