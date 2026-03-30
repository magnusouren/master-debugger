import asyncio
from pathlib import Path
import matplotlib.pyplot as plt

from backend.types.config import SystemConfig, OperationMode
from backend.core.runtime_controller import RuntimeController
from backend.analysis.plot_experiments import _parse_estimates, plot_estimates
from backend.services.eye_tracker.tobii_replay_adapter import TobiiReplayAdapter
from backend.services.logger_service import get_logger


CONFIG_PATH = "backend/config.yaml"

REPLAY_DIR = Path("logs/record")

OUTPUT_DIR = Path("logs/figures")


def build_config(replay_file: Path, config_path: str) -> SystemConfig:

    cfg = SystemConfig.from_file(config_path)

    cfg.eye_tracker.mode = "TOBII_REPLAY"

    cfg.eye_tracker.filepath = str(replay_file)

    # Start in reactive so baseline calibration runs first.
    cfg.controller.operation_mode = OperationMode.REACTIVE

    cfg.signal_processing.min_valid_sample_ratio = 0.0

    cfg.forecasting.update_rate_hz = 0.0

    cfg.forecasting.min_confidence_threshold = 0.0

    # Keep replay paced so calibration has real data at the beginning.
    cfg.eye_tracker.fast_forward = False

    return cfg


async def run_single(
    replay_file: Path,
    config_path: str,
    output_dir: Path
):

    experiment_id = replay_file.stem

    participant_id = replay_file.stem

    get_logger().reset()

    cfg = build_config(replay_file, config_path)

    rc = RuntimeController(cfg)

    await rc.initialize()

    try:

        if isinstance(rc._eye_tracker_adapter, TobiiReplayAdapter):

            rc._eye_tracker_adapter._fast_forward = False

            rc._eye_tracker_adapter._batch_size = 500

            rc._eye_tracker_adapter._flush_interval_ms = 1

        if not await rc.connect_eye_tracker():

            print(f"[WARN] Could not connect replay {replay_file}")

            return None

        await rc.start_experiment(
            experiment_id,
            participant_id
        )

        # Wait for automatic baseline calibration (5s settling + configured duration),
        # then switch to proactive mode for the remaining replay.
        calibration_wait_seconds = 5.0 + max(0.0, cfg.controller.calibration_duration_seconds)
        await asyncio.sleep(calibration_wait_seconds)
        if rc.get_operation_mode() != OperationMode.PROACTIVE:
            rc.set_operation_mode(OperationMode.PROACTIVE)

        # Wait for replay to finish

        adapter = rc._eye_tracker_adapter

        if adapter and getattr(adapter, "_stream_task", None):

            await adapter._stream_task

        # Give runtime time to process last windows.
        await asyncio.sleep(2.0)

        await rc.end_experiment()

    finally:

        await rc.shutdown()

    exp_dir = Path("logs/experiments")

    pattern = f"*{participant_id}_{experiment_id}*.csv"

    matches = sorted(
        exp_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime
    )

    if not matches:

        print(f"[WARN] No experiment log for {replay_file}")

        return None

    csv_path = matches[-1]

    df, trigger_bounds, feedback_interactions = _parse_estimates(csv_path)

    if df.empty:

        print(f"[WARN] No estimates in log")

        return csv_path

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig = plot_estimates(
        df,
        title=f"{experiment_id}",
        trigger_bounds=trigger_bounds,
        feedback_interactions=feedback_interactions,
    )

    outfile = output_dir / f"plot_{experiment_id}.png"

    fig.savefig(outfile, dpi=150)

    plt.close(fig)

    print(f"Saved plot {outfile}")

    return csv_path


async def main(
    config_path: str = CONFIG_PATH,
    replay_dir: Path = REPLAY_DIR,
    output_dir: Path = OUTPUT_DIR
):

    files = sorted(
        p for p in replay_dir.glob("*.tsv")
        if p.is_file()
    )

    if not files:

        print(f"No TSV files found in {replay_dir}")

        return

    for f in files:

        print(f"\n=== Replaying {f.name} ===")

        await run_single(
            f,
            config_path,
            output_dir
        )


if __name__ == "__main__":

    asyncio.run(
        main(
            CONFIG_PATH,
            REPLAY_DIR,
            OUTPUT_DIR
        )
    )