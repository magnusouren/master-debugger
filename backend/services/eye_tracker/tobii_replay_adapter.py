"""
Tobii Replay Eye Tracker Adapter

Replays runtime-recorded Tobii gaze samples from TSV.
Produces identical GazeSample objects as TobiiProEyeTrackerAdapter.

Supports:
- real-time replay
- fast forward
- deterministic timestamps
"""

import asyncio
import csv
import time
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

from backend.services.eye_tracker.base import EyeTrackerAdapter, AdapterState
from backend.types.eye_tracking import GazeSample
from backend.services.logger_service import get_logger


class TobiiReplayAdapter(EyeTrackerAdapter):

    def __init__(
        self,
        file_path: str,
        batch_size: int = 12,
        flush_interval_ms: int = 16,
        fast_forward: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):

        self._file_path = Path(file_path)

        self._batch_size = batch_size
        self._flush_interval_ms = flush_interval_ms

        self._fast_forward = fast_forward

        self._loop = loop or asyncio.get_event_loop()

        self._state = AdapterState.DISCONNECTED

        self._samples: List[GazeSample] = []

        self._buffer: List[GazeSample] = []

        self._samples_callback: Optional[
            Callable[[List[GazeSample]], None]
        ] = None

        self._error_callback: Optional[
            Callable[[Exception], None]
        ] = None

        self._stream_task: Optional[asyncio.Task] = None

        self._logger = get_logger()

    async def connect(self, device_id: Optional[str] = None) -> bool:

        if self._state != AdapterState.DISCONNECTED:
            return False

        try:

            self._samples = self._load_samples()

            if not self._samples:
                raise ValueError("No samples in replay file")

            self._state = AdapterState.CONNECTED

            return True

        except Exception as e:

            self._state = AdapterState.ERROR

            if self._error_callback:
                self._error_callback(e)

            return False

    async def disconnect(self):

        if self._state == AdapterState.STREAMING:
            await self.stop_streaming()

        self._samples.clear()
        self._buffer.clear()

        self._state = AdapterState.DISCONNECTED

    def is_connected(self):

        return self._state in (
            AdapterState.CONNECTED,
            AdapterState.STREAMING,
        )

    def set_samples_callback(self, callback):

        self._samples_callback = callback

    def set_error_callback(self, callback):

        self._error_callback = callback

    async def start_streaming(self):

        if self._state != AdapterState.CONNECTED:
            raise RuntimeError("Adapter not connected")

        self._state = AdapterState.STREAMING

        self._stream_task = asyncio.create_task(
            self._stream()
        )

    async def stop_streaming(self):

        if self._stream_task:

            self._stream_task.cancel()

            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

            self._stream_task = None

        if self._buffer and self._samples_callback:
            self._samples_callback(self._buffer.copy())

        self._buffer.clear()

        self._state = AdapterState.CONNECTED

    def get_state(self):

        return self._state

    def get_device_info(self):

        return {

            "device_id": f"replay:{self._file_path.name}",

            "model": "Tobii Replay Adapter",

            "source": str(self._file_path),

            "sample_count": len(self._samples),

            "fast_forward": self._fast_forward,
        }

    async def _stream(self):

        flush_interval = self._flush_interval_ms / 1000.0

        last_flush = time.time()

        try:

            prev_ts = None

            for sample in self._samples:

                if self._state != AdapterState.STREAMING:
                    break

                if not self._fast_forward:

                    if prev_ts is not None:

                        dt = sample.timestamp - prev_ts

                        dt = max(0.001, min(dt, 0.1))

                        await asyncio.sleep(dt)

                prev_ts = sample.timestamp

                self._buffer.append(sample)

                now = time.time()

                if (
                    len(self._buffer) >= self._batch_size
                    or (now - last_flush) >= flush_interval
                ):

                    if self._samples_callback:

                        batch = self._buffer.copy()

                        self._buffer.clear()

                        self._samples_callback(batch)

                    last_flush = now

                    if self._fast_forward:
                        await asyncio.sleep(0)

            if self._buffer and self._samples_callback:

                self._samples_callback(self._buffer.copy())

                self._buffer.clear()

            self._state = AdapterState.CONNECTED

        except Exception as e:

            self._state = AdapterState.ERROR

            if self._error_callback:
                self._error_callback(e)

            raise

    def _load_samples(self):

        samples = []

        with self._file_path.open(
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(
                f,
                delimiter="\t"
            )

            for row in reader:

                try:

                    device_ts = float(
                        row["device_timestamp_us"]
                    ) / 1_000_000.0

                    timestamp = float(
                        row["system_timestamp_s"]
                    )

                    sample = GazeSample(

                        timestamp=timestamp,

                        left_eye_x=self._to_float(row["left_x_norm"]),

                        left_eye_y=self._to_float(row["left_y_norm"]),

                        right_eye_x=self._to_float(row["right_x_norm"]),

                        right_eye_y=self._to_float(row["right_y_norm"]),

                        left_pupil_diameter=self._to_float(
                            row["left_pupil_mm"]
                        ),

                        right_pupil_diameter=self._to_float(
                            row["right_pupil_mm"]
                        ),

                        left_eye_valid=row["left_valid"] == "1",

                        right_eye_valid=row["right_valid"] == "1",

                        raw_data={
                            "device_time_stamp":
                            int(device_ts * 1_000_000),

                            "source": "tobii_replay"
                        }
                    )

                    samples.append(sample)

                except Exception as e:

                    self._logger.system(
                        "replay_row_failed",
                        {"error": str(e)},
                        level="WARNING"
                    )

        return samples

    def _to_float(self, v):

        if v is None or v == "":
            return None

        return float(v)
