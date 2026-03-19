import asyncio
from pathlib import Path
import matplotlib.pyplot as plt
from backend.types.config import SystemConfig, OperationMode
from backend.core.runtime_controller import RuntimeController
from backend.analysis.plot_experiments import _select_files, _parse_estimates, plot_estimates

CONFIG_PATH = "backend/config.yaml"
REPLAY_FILE = "logs/replay/202_rawdata.tsv"
EXPERIMENT_ID = "replay_run"
PARTICIPANT_ID = "replay_user"

async def main():
    # Load and tweak config
    cfg = SystemConfig.from_file(CONFIG_PATH)
    cfg.eye_tracker.mode = "REPLAY"
    cfg.eye_tracker.filepath = REPLAY_FILE
    cfg.controller.operation_mode = OperationMode.PROACTIVE
    cfg.controller.calibration_duration_seconds = 0  # skip baseline scheduling entirely
    # Keep sampling rate aligned with replay file (250 Hz)
    cfg.signal_processing.input_sampling_rate_hz = 250.0

    # Replay adapter overrides for speed
    cfg.eye_tracker.mode = "REPLAY"
    # Fast-forward: emit samples as fast as possible. Use synthetic/system timestamps to avoid large gaps from the file.
    cfg.eye_tracker.replay_fast_forward = True  # custom field consumed below
    cfg.eye_tracker.replay_use_system_timestamps = True
    # For fast replay, loosen quality threshold heavily (emit all windows)
    cfg.signal_processing.min_valid_sample_ratio = 0.0
    # Allow predictions every window during fast replay
    cfg.forecasting.update_rate_hz = 0.0  # no throttling
    cfg.forecasting.min_confidence_threshold = 0.0

    # Inject adapter kwargs for fast replay
    rc = RuntimeController(cfg)
    # After initialize, grab the adapter and force fast-forward settings
    from backend.services.eye_tracker.replay_adapter import ReplayEyeTrackerAdapter
    await rc.initialize()

    if isinstance(rc._eye_tracker_adapter, ReplayEyeTrackerAdapter):
        rc._eye_tracker_adapter._fast_forward = True
        rc._eye_tracker_adapter._use_system_timestamps = True
        # Optional: also tighten batch/flush for fewer callbacks
        rc._eye_tracker_adapter._batch_size = 50
        rc._eye_tracker_adapter._flush_interval_ms = 2

    if not await rc.connect_eye_tracker():
        raise SystemExit("Eye tracker connect failed")

    # Start experiment (this also starts streaming on the replay adapter)
    await rc.start_experiment(EXPERIMENT_ID, PARTICIPANT_ID)

    # Wait until the replay adapter runs out of samples (or processing finishes)
    adapter = rc._eye_tracker_adapter  # owned by controller; safe to await its streaming task
    if adapter and getattr(adapter, "_streaming_task", None):
        await adapter._streaming_task

    # Give processing time to flush windows and estimates after fast-forward
    await asyncio.sleep(2.0)

    # End experiment and shut down
    await rc.end_experiment()
    await rc.shutdown()

    # Plot latest experiment log
    log_dir = Path("logs/experiments")
    latest_csv = _select_files(log_dir, "experiment_*.csv", session_id=None, files=None, include_all=False)[0]
    df = _parse_estimates(latest_csv)
    if df.empty:
        print("No estimates found in log; nothing to plot.")
        return

    fig = plot_estimates(df, title=f"{EXPERIMENT_ID} ({PARTICIPANT_ID})")

    output_dir = Path("logs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"plot_{EXPERIMENT_ID}_{PARTICIPANT_ID}.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")

if __name__ == "__main__":
    asyncio.run(main())