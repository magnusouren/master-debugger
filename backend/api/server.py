"""
Combined Server

Main server that runs both WebSocket and REST API servers together.
Entry point for the backend application.
"""
import datetime
from typing import Optional, Dict, Any
import asyncio
import signal

from backend.types import SystemConfig
from backend.layers import RuntimeController
from backend.api.websocket_server import WebSocketServer
from backend.api.rest_api import RestAPI
from backend.types.code_context import CodeContext
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
        self._loop = asyncio.get_event_loop()
        self._setup_signal_handlers()
        
        try:
            self._loop.run_until_complete(self.start())
            self._loop.run_forever()
        except KeyboardInterrupt:
            print("[Server] Keyboard interrupt received. Shutting down...")
        finally:
            self._loop.run_until_complete(self.stop())
            self._loop.close()
    
    async def run_async(self) -> None:
        """Run the server asynchronously."""
        await self.start()
    
    def is_running(self) -> bool:
        """
        Check if server is running.
        
        Returns:
            True if server is running.
        """
        return self._is_running
    
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
        if self._loop is None:
            return
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._shutdown_handler(s))
            )
        
    
    def _wire_components(self) -> None:
    # Controller -> WebSocket (outbound)
        async def send_outbound(msg: WebSocketMessage) -> None:
            # broadcast by default; you can add targeting later

            # target specific clients if needed
            if msg.target_client_id:
                try:
                    print(f"[Server] Sending outbound message to client {msg.target_client_id} via WebSocket")
                    await self._websocket_server.send_to_client(msg.target_client_id, msg)
                    return
                except Exception as e:
                    print(f"[Server] Error sending outbound message to client {msg.target_client_id} via WebSocket: {e}")
                    return
                
            # broadcast to all clients
            try:
                await self._websocket_server.broadcast(msg)
            except Exception as e:
                print(f"[Server] Error broadcasting outbound message via WebSocket: {e}")

        self._controller.register_websocket_callback(send_outbound)

        # WebSocket -> Controller (inbound)
        self._setup_websocket_handlers()

        # REST routes -> Controller (inbound)
        self._setup_api_routes()
    
    def _setup_api_routes(self) -> None:
        """Set up REST API routes with controller handlers."""
        pass  # TODO: Implement route setup
    
    def _setup_websocket_handlers(self) -> None:
        """
        Set up WebSocket message handlers to handle messages correctly.
        """
        async def on_context_update(message: WebSocketMessage, client_id: str) -> None:
            print(f"[Server] Received context update from client {client_id}")

            # Convert payload into your internal CodeContext type
            ctx = CodeContext.from_dict(message.payload) 
            ctx.metadata = {**ctx.metadata, "client_id": client_id}
            await self._controller.handle_context_update(ctx)

        async def on_ping(message: WebSocketMessage, client_id: str) -> None:
            pong_msg = WebSocketMessage(
                type=MessageType.PONG,
                timestamp=datetime.datetime.now(datetime.timezone.utc).timestamp(),
                payload={},
                message_id=message.message_id,  # Echo the incoming message_id
            )
            await self._websocket_server.send_to_client(client_id, pong_msg)

        # Register handlers here

        # Register the handler for context updates
        self._websocket_server.register_handler(MessageType.CONTEXT_UPDATE, on_context_update)
        self._websocket_server.register_handler(MessageType.PING, on_ping)
    
    async def _shutdown_handler(self, sig: signal.Signals) -> None:
        """
        Handle shutdown signals.
        
        Args:
            sig: The signal received.
        """
        print(f"[Server] Received shutdown signal: {sig.name}. Shutting down...")
        await self.stop()
        self._loop.stop()


def create_server(config_path: Optional[str] = None) -> Server:
    """
    Factory function to create a server instance.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        Configured server instance.
    """
    config = SystemConfig.load_from_file(config_path) if config_path else SystemConfig()
    server = Server(config)
    return server


def main() -> None:
    """Main entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the combined backend server.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    server = create_server(args.config)
    server.run()


if __name__ == "__main__":
    main()
