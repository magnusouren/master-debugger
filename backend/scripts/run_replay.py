import asyncio
from pathlib import Path
import matplotlib.pyplot as plt
from backend.types.config import SystemConfig, OperationMode
from backend.core.runtime_controller import RuntimeController
from backend.analysis.plot_experiments import _parse_estimates, plot_estimates
from backend.services.eye_tracker.replay_adapter import ReplayEyeTrackerAdapter
from backend.services.logger_service import get_logger

CONFIG_PATH = "backend/config.yaml"
REPLAY_DIR = Path("logs/replay")
OUTPUT_DIR = Path("logs/figures")


def build_config(replay_file: Path, config_path: str) -> SystemConfig:
    cfg = SystemConfig.from_file(config_path)
    cfg.eye_tracker.mode = "REPLAY"
    cfg.eye_tracker.filepath = str(replay_file)
    cfg.controller.operation_mode = OperationMode.PROACTIVE
    cfg.controller.calibration_duration_seconds = 0  # skip baseline
    cfg.signal_processing.input_sampling_rate_hz = 250.0
    cfg.signal_processing.min_valid_sample_ratio = 0.0
    cfg.forecasting.update_rate_hz = 0.0
    cfg.forecasting.min_confidence_threshold = 0.0
    # Custom flags consumed directly by adapter instance tweaks below
    cfg.eye_tracker.replay_fast_forward = True
    cfg.eye_tracker.replay_use_system_timestamps = True
    return cfg


async def run_single(replay_file: Path, config_path: str, output_dir: Path) -> Path | None:
    experiment_id = replay_file.stem
    participant_id = replay_file.stem

    # Ensure logger buffers are cleared so each run starts clean
    get_logger().reset()

    cfg = build_config(replay_file, config_path)
    rc = RuntimeController(cfg)
    await rc.initialize()

    try:
        if isinstance(rc._eye_tracker_adapter, ReplayEyeTrackerAdapter):
            rc._eye_tracker_adapter._fast_forward = True
            rc._eye_tracker_adapter._use_system_timestamps = True
            rc._eye_tracker_adapter._batch_size = 100
            rc._eye_tracker_adapter._flush_interval_ms = 1

        if not await rc.connect_eye_tracker():
            print(f"[WARN] Could not connect replay for {replay_file}")
            return None

        await rc.start_experiment(experiment_id, participant_id)

        adapter = rc._eye_tracker_adapter
        if adapter and getattr(adapter, "_streaming_task", None):
            await adapter._streaming_task

        await rc.wait_for_background_tasks()

        await rc.end_experiment()
    finally:
        await rc.shutdown()

    # Locate newest experiment CSV for this run
    exp_dir = Path("logs/experiments")
    pattern = f"*{participant_id}_{experiment_id}*.csv"
    matches = sorted(exp_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        print(f"[WARN] No experiment log for {replay_file}")
        return None

    csv_path = matches[-1]
    df, trigger_bounds, feedback_interactions = _parse_estimates(csv_path)
    if df.empty:
        print(f"[WARN] No estimates in log for {replay_file}")
        return csv_path

    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_estimates(
        df,
        title=f"{experiment_id}",
        trigger_bounds=trigger_bounds,
        feedback_interactions=feedback_interactions,
    )
    outfile = output_dir / f"plot_{experiment_id}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {outfile}")
    return csv_path


async def main(config_path: str = CONFIG_PATH, replay_dir: Path = REPLAY_DIR, output_dir: Path = OUTPUT_DIR):
    files = sorted(p for p in replay_dir.glob("*.tsv") if p.is_file())
    if not files:
        print(f"No TSV files found in {replay_dir}")
        return

    for f in files:
        print(f"\n=== Replaying {f.name} ===")
        await run_single(f, config_path, output_dir)


if __name__ == "__main__":
    asyncio.run(main(CONFIG_PATH, REPLAY_DIR, OUTPUT_DIR))
