#!/usr/bin/env python3
"""
Eye Tracking Debugger Backend - Main Entry Point

Usage:
    python -m backend.main [--config CONFIG_PATH] [--host HOST] [--port PORT]
    
Or after installing the package:
    eye-tracking-backend [--config CONFIG_PATH] [--host HOST] [--port PORT]
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Eye Tracking Debugger Backend Server"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind servers to",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8080,
        help="REST API server port",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["reactive", "proactive"],
        default="reactive",
        help="Operation mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def load_config(config_path: str | None) -> "SystemConfig":
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        System configuration.
    """
    # TODO: Implement configuration loading
    from backend.types import SystemConfig
    return SystemConfig()


def setup_logging(debug: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        debug: Enable debug level logging.
    """
    import logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


async def run_server(config: "SystemConfig") -> None:
    """
    Run the backend server.
    
    Args:
        config: System configuration.
    """
    from backend.api.server import Server
    import signal
    
    server = Server(config)
    
    print("Starting servers...")
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        print("\nShutdown signal received...")
        shutdown_event.set()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    # Keep the server running until interrupted
    try:
        await server.start()
        # Wait for shutdown signal
        await shutdown_event.wait()
    except asyncio.CancelledError:
        print("Server shutdown requested")
    finally:
        print("Stopping servers...")
        await server.stop()
        print("Servers stopped gracefully")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set up logging
    setup_logging(debug=args.debug)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    config.controller.websocket_host = args.host
    config.controller.websocket_port = args.ws_port
    config.controller.api_host = args.host
    config.controller.api_port = args.api_port
    
    # TODO: Set operation mode from args
    
    print(f"Starting Eye Tracking Debugger Backend...")
    print(f"  WebSocket: ws://{args.host}:{args.ws_port}")
    print(f"  REST API:  http://{args.host}:{args.api_port}")
    print(f"  Mode: {args.mode}")
    
    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
