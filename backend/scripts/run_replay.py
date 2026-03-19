import asyncio
from pathlib import Path
import matplotlib.pyplot as plt
from backend.types.config import SystemConfig, OperationMode
from backend.core.runtime_controller import RuntimeController
from backend.analysis.plot_experiments import _select_files, _parse_estimates, plot_estimates

CONFIG_PATH = "backend/config.yaml"
REPLAY_FILE = "logs/replay/207_rawdata.tsv"
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

    rc = RuntimeController(cfg)
    await rc.initialize()

    if not await rc.connect_eye_tracker():
        raise SystemExit("Eye tracker connect failed")

    # Start experiment (this also starts streaming on the replay adapter)
    await rc.start_experiment(EXPERIMENT_ID, PARTICIPANT_ID)

    # Wait until the replay adapter runs out of samples
    adapter = rc._eye_tracker_adapter  # owned by controller; safe to await its streaming task
    if adapter and getattr(adapter, "_streaming_task", None):
        await adapter._streaming_task

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
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())