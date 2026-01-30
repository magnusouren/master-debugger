# Eye Tracker Adapter Service

This module provides adapter interfaces and implementations for integrating eye tracker hardware into the system.

## Architecture

The eye tracker adapter service follows a clean architecture pattern with:

1. **Base Interface** (`base.py`): Defines the `EyeTrackerAdapter` protocol that all implementations must follow
2. **Implementations**: 
   - `SimulatedEyeTrackerAdapter`: Stub implementation for testing without hardware
   - `TobiiProEyeTrackerAdapter`: Real hardware integration with Tobii Pro SDK
3. **Factory** (`factory.py`): Creates appropriate adapter based on configuration

## Key Features

### Batched Sample Processing
- Eye trackers like Tobii Pro X3-120 stream data at ~120 Hz
- Adapter batches samples efficiently before forwarding to signal processing
- Default: batch_size=12, flush_interval_ms=16
- Reduces overhead and handles bursty callback behavior

### Thread-Safe Design
- Tobii SDK callbacks occur on non-asyncio threads
- Uses `loop.call_soon_threadsafe` to forward data into asyncio loop
- Buffer operations are protected with locks

### State Management
- Clear state transitions: DISCONNECTED → CONNECTING → CONNECTED → STREAMING
- Error state for fault handling
- Clean cancellation of streaming tasks

### No Hard Dependencies
- Tobii Pro SDK is lazily imported only when needed
- System works without SDK installed (falls back to simulated mode)
- RuntimeController never imports Tobii SDK directly

## Usage

### Configuration

Add to your `config.yaml`:

```yaml
controller:
  # Eye Tracker Settings
  eye_tracker_mode: SIMULATED  # SIMULATED or TOBII
  eye_tracker_device_id: null  # null for auto-select, or specific device serial/address
  eye_tracker_batch_size: 12  # Number of samples per batch
  eye_tracker_flush_interval_ms: 16  # Max time between flushes
```

### Connecting to Eye Tracker

```python
# In RuntimeController
await controller.connect_eye_tracker()  # Uses config settings
# or
await controller.connect_eye_tracker(device_id="specific-device-id")
```

### Disconnecting

```python
await controller.disconnect_eye_tracker()
```

## Implementation Details

### SimulatedEyeTrackerAdapter

- Generates synthetic gaze data at configurable rate
- Simulates realistic eye movements and pupil diameter changes
- Useful for testing and development

### TobiiProEyeTrackerAdapter

- Discovers and connects to Tobii Pro eye trackers
- Subscribes to gaze data stream
- Converts Tobii SDK format to internal `GazeSample` format
- Handles device info retrieval (model, serial, sampling rate)

#### Installing Tobii Pro SDK

```bash
pip install tobii-research
```

## Data Flow

```
Eye Tracker Hardware
  ↓ (SDK callbacks, ~120 Hz)
TobiiProEyeTrackerAdapter
  ↓ (batching & thread-safe forwarding)
RuntimeController._on_gaze_samples()
  ↓ (batch of GazeSample objects)
SignalProcessingLayer.add_samples()
```

## Error Handling

- SDK import failures are caught and logged
- Connection failures return False (no crash)
- Streaming errors trigger error callback and set ERROR state
- Clean shutdown ensures resources are released

## Testing

You can test the system without hardware:

1. Set `eye_tracker_mode: SIMULATED` in config
2. Start the system normally
3. Simulated adapter will generate realistic sample data

## Extension

To add a new eye tracker:

1. Create a new adapter class implementing `EyeTrackerAdapter`
2. Add factory logic in `factory.py`
3. Add configuration option in `ControllerConfig`
4. No changes needed to RuntimeController!
