"""
Combined Server

Main server that runs both WebSocket and REST API servers together.
Entry point for the backend application.
"""
from typing import Optional, Dict, Any
import asyncio
import signal

from backend.types import SystemConfig
from backend.layers import RuntimeController
from backend.api.websocket_server import WebSocketServer
from backend.api.rest_api import RestAPI
from backend.types.code_context import CodeContext
from backend.types.feedback import FeedbackInteraction
from backend.types.messages import MessageType, WebSocketMessage


class Server:
    """
    Main server combining WebSocket and REST API.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the combined server.
        
        Args:
            config: System configuration.
        """
        self._config = config or SystemConfig()
        
        # Initialize components
        self._controller = RuntimeController(self._config)
        self._websocket_server = WebSocketServer(
            host=self._config.controller.websocket_host,
            port=self._config.controller.websocket_port,
        )
        self._rest_api = RestAPI(
            host=self._config.controller.api_host,
            port=self._config.controller.api_port,
        )
        
        self._is_running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None



    
    async def start(self) -> None:
        """Start all server components."""
        print(f"[Server] Starting WebSocket server on ws://{self._config.controller.websocket_host}:{self._config.controller.websocket_port}")
        print(f"[Server] Starting REST API server on http://{self._config.controller.api_host}:{self._config.controller.api_port}")
        
        # Wire up the components
        self._wire_components()

        # Start WebSocket and REST API servers
        await self._websocket_server.start()
        await self._rest_api.start()
        
        # Initialize the controller
        await self._controller.initialize()
        
        self._is_running = True
        print("[Server] All servers started successfully! \n")
    
    async def stop(self) -> None:
        """Stop all server components gracefully."""
        print("[Server] Stopping servers...")
        self._is_running = False
        
        await self._controller.shutdown()
        await self._websocket_server.stop()
        await self._rest_api.stop()
        
        print("[Server] All servers stopped")
    
    def run(self) -> None:
        """
        Run the server (blocking).
        
        This is the main entry point for running the server.
        """
        pass  # TODO: Implement blocking run
    
    async def run_async(self) -> None:
        """Run the server asynchronously."""
        pass  # TODO: Implement async run
    
    def is_running(self) -> bool:
        """
        Check if server is running.
        
        Returns:
            True if server is running.
        """
        pass  # TODO: Implement status check
    
    def get_controller(self) -> RuntimeController:
        """
        Get the runtime controller instance.
        
        Returns:
            The runtime controller.
        """
        return self._controller
    
    def get_websocket_server(self) -> WebSocketServer:
        """
        Get the WebSocket server instance.
        
        Returns:
            The WebSocket server.
        """
        return self._websocket_server
    
    def get_rest_api(self) -> RestAPI:
        """
        Get the REST API instance.
        
        Returns:
            The REST API server.
        """
        return self._rest_api
    
    # --- Internal Methods ---
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        pass  # TODO: Implement signal handlers
    
    def _wire_components(self) -> None:
    # Controller -> WebSocket (outbound)
        async def send_outbound(msg: WebSocketMessage) -> None:
            # broadcast by default; you can add targeting later
            await self._websocket_server.broadcast(msg)

        self._controller.register_websocket_callback(send_outbound)

        # WebSocket -> Controller (inbound)
        self._setup_websocket_handlers()

        # REST routes -> Controller (inbound)
        self._setup_api_routes()
    
    def _setup_api_routes(self) -> None:
        """Set up REST API routes with controller handlers."""
        pass  # TODO: Implement route setup
    
    def _setup_websocket_handlers(self) -> None:
        async def on_context_update(message: WebSocketMessage, client_id: str) -> None:
            print(f"[Server] Received context update from client {client_id}")

            # Convert payload into your internal CodeContext type
            ctx = CodeContext.from_dict(message.payload) 
            await self._controller.handle_context_update(ctx)

        # Register the handler for context updates
        self._websocket_server.register_handler(MessageType.CONTEXT_UPDATE, on_context_update)
    
    async def _shutdown_handler(self, sig: signal.Signals) -> None:
        """
        Handle shutdown signals.
        
        Args:
            sig: The signal received.
        """
        pass  # TODO: Implement shutdown handler


def create_server(config_path: Optional[str] = None) -> Server:
    """
    Factory function to create a server instance.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        Configured server instance.
    """
    pass  # TODO: Implement server factory


def main() -> None:
    """Main entry point for running the server."""
    pass  # TODO: Implement main function


if __name__ == "__main__":
    main()
