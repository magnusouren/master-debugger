# Eye Tracking Debugger

A system for providing eye-tracking based debugging assistance in VS Code.

## Project Structure

TODO

## Requirements

### Backend

- Python 3.11 (required for tobii-research SDK)
- Tobii Eye Tracker

# Installations

libomp (for xgboost on macOS):

```bash
brew install libomp
```

### VS Code Extension

- Node.js 18+
- VS Code 1.85+

## Training Data

The forecasting model in this project was trained using the EMIP eye-tracking dataset available on OSF: https://osf.io/53kts/overview.

## Setup

### Backend

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (from repo root)
pip install -r backend/requirements.txt

# Run the backend (from root)
python -m backend.main --config backend/config.yaml

```

The backend will start a WebSocket server on port 8765 and a REST API server on port 8080 by default. Adjust ports in `config.yaml` as needed.

### Training (XGBoost forecaster)

```bash

# Train model (uses participant-level train/val/test split)
python -m backend.training.train_xgboost --config backend/config.yaml
```

The active proactive model is loaded from `backend/models/trained/latest.json` with metadata in `backend/models/trained/latest_metadata.json`.

### VS Code Extension

```bash
cd vscode-extension

# Install dependencies
npm install

# Compile
npm run compile

# For development, use watch mode
npm run watch
```

To test the extension:

1. Open the `vscode-extension` folder in VS Code
2. Press `F5` to launch the Extension Development Host
3. Use the command palette to run Eye Tracking commands

### Usage per now (30th of January 2026)

1. Start the backend server
2. Launch the VS Code extension
3. Connect the extension to the backend via the command palette ("Eye Tracking: Connect to Backend")
4. Connect the eye tracker via the command palette ("Eye Tracking: Connect Eye Tracker")
5. Open a code file in the editor
6. Trigger feedback generation ("Eye Tracking: Trigger Feedback Send")

## Architecture

### Data Flow

1. **Eye Tracker → Signal Processing**: Raw gaze data (120 Hz) is processed into window-based features (2-10 Hz)

2. **Reactive mode**: Signal Processing → Reactive Tool (direct scoring)

3. **Proactive mode**: Signal Processing → Forecasting Tool (predict +30s) → Reactive Tool (baseline-aware scoring)

4. **Reactive Tool → Controller**: User state scores trigger feedback decisions

5. **Controller → Feedback Layer → VS Code**: Structured feedback is generated and rendered in the editor

### Operation Modes

- **Reactive**: Responds to current user state in real-time
- **Proactive**: Uses forecasting to predict future states and provide preemptive assistance

## Configuration

Copy `backend/config.example.yaml` to `backend/config.yaml` and customize:

- Signal processing parameters (window size, metrics, output frequency)
- Forecasting settings (prediction horizon, model path)
- Reactive tool thresholds
- Feedback layer LLM settings
- Server ports and logging

## Declaration of AI:

This project has been developed with the assistance of AI tools to enhance code quality and documentation. All code and content have been reviewed and validated by the project maintainers to ensure accuracy and integrity.
