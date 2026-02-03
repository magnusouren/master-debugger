# VS Code Extension

This is the VS Code extension frontend for the Eye Tracking Debugger system.

## Development

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Setup

```bash
cd vscode-extension
npm install
```

### Build

```bash
npm run compile
```

### Watch mode

```bash
npm run watch
```

### Testing

1. Press `F5` in VS Code to open a new Extension Development Host window
2. Use the command palette (`Cmd+Shift+P` / `Ctrl+Shift+P`) to run Eye Tracking commands

## Commands

- `Eye Tracking: Connect to Backend` - Connect to the Python backend
- `Eye Tracking: Disconnect from Backend` - Disconnect from the backend
- `Eye Tracking: Toggle Reactive/Proactive Mode` - Switch operation modes
- `Eye Tracking: Show Status` - Show current system status
- `Eye Tracking: Clear All Feedback` - Clear displayed feedback

## Configuration

Configure in VS Code settings:

- `eyeTrackingDebugger.backendHost`: Backend server host (default: "localhost")
- `eyeTrackingDebugger.websocketPort`: WebSocket port (default: 8765)
- `eyeTrackingDebugger.apiPort`: REST API port (default: 8080)
- `eyeTrackingDebugger.autoConnectBackend`: Auto-connect on startup (default: false)
- `eyeTrackingDebugger.autoConnectEyeTracker`: Auto-connect eye tracker on startup (default: false)
- `eyeTrackingDebugger.showInlineHints`: Show inline hints (default: true)
