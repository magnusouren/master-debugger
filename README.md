# Eye Tracking Debugger

A system for providing eye-tracking based debugging assistance in VS Code.

## Project Structure

```
master-debuger/
├── backend/                            # Python backend
│   ├── api/                            # WebSocket and REST API servers
│   │   ├── __init__.py 
│   │   ├── websocket_server.py         # Real-time communication
│   │   ├── rest_api.py                 # Configuration and status endpoints
│   │   └── server.py                   # Combined server entry point
│   ├── layers/                         # Processing layers
│   │   ├── __init__.py 
│   │   ├── signal_processing.py        # Raw gaze data → features
│   │   ├── forecasting_tool.py         # Feature prediction (proactive)
│   │   ├── reactive_tool.py            # User state estimation
│   │   ├── feedback_layer.py           # Feedback generation
│   │   └── runtime_controller.py       # Central orchestrator
│   ├── services/                       # Service implementations
│   │   ├── eye_tracker/                # Eye tracker interfaces
│   │   │   ├── tobii_pro_adapter.py    # Tobii Pro SDK adapter
│   │   │   └── simulated_tracker.py    # Simulated eye tracker for testing
│   │   ├── llm_clients/                # LLM service clients
│   │   │   ├── openai_client.py        # OpenAI API client
│   │   │   └── development.py          # Anthropic API client
│   │   └── logger.py                   # Logging service
│   ├── types/                          # Type definitions
│   │   ├── __init__.py     
│   │   ├── eye_tracking.py             # Eye tracking data types
│   │   ├── user_state.py               # User state types
│   │   ├── code_context.py             # VS Code context types
│   │   ├── feedback.py                 # Feedback types
│   │   ├── config.py                   # Configuration types
│   │   └── messages.py                 # WebSocket message types
│   ├── __init__.py     
│   ├── main.py                         # Main entry point
│   ├── requirements.txt                # Python dependencies
│   └── config.example.yaml             # Example configuration
├── vscode-extension/                   # VS Code extension frontend
│   ├── src/        
│   │   ├── extension.ts                # Extension entry point
│   │   ├── types.ts                    # TypeScript type definitions
│   │   ├── websocket-client.ts         # WebSocket client
│   │   ├── context-collector.ts        # Editor context collection
│   │   ├── feedback-renderer.ts        # Feedback display
│   │   └── status-bar.ts               # Status bar management
│   ├── package.json    
│   ├── tsconfig.json   
│   └── README.md   
├── .venv/                              # Python virtual environment
└── README.md                           # This file
```

## Requirements

### Backend
- Python 3.11 (required for tobii-research SDK)
- Tobii Eye Tracker

### VS Code Extension
- Node.js 18+
- VS Code 1.85+

## Setup

### Backend

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Run the backend
cd backend
python main.py --config config.yaml
```
The backend will start a WebSocket server on port 8765 and a REST API server on port 8080 by default. Adjust ports in `config.yaml` as needed.

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

2. **Signal Processing → Reactive Tool**: Features are analyzed to estimate user state (stress, cognitive load, etc.)

3. **Reactive Tool → Controller**: User state scores trigger feedback decisions

4. **Controller → Feedback Layer**: When thresholds are exceeded, feedback is generated using code context

5. **Feedback Layer → VS Code**: Structured feedback is sent via WebSocket and rendered in the editor

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