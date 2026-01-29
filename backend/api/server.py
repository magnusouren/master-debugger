"""
Combined Server

Main server that runs both WebSocket and REST API servers together.
Entry point for the backend application.
"""
import datetime
from typing import Optional, Dict, Any
import asyncio
import signal

from backend.api.serialization import json_safe
from backend.types import SystemConfig
from backend.layers import RuntimeController
from backend.api.websocket_server import WebSocketServer
from backend.api.rest_api import HttpMethod, RestAPI
from backend.services.logger_service import get_logger
from backend.types.code_context import CodeContext
from backend.types.messages import MessageType, WebSocketMessage
from backend.types.domain_events import DomainEvent, DomainEventType


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
        self._logger = get_logger()



    
    async def start(self) -> None:
        """Start all server components."""
        self._logger.system(
            "servers_starting",
            {
                "websocket_url": f"ws://{self._config.controller.websocket_host}:{self._config.controller.websocket_port}",
                "api_url": f"http://{self._config.controller.api_host}:{self._config.controller.api_port}",
            },
        )
        
        # Wire up the components
        self._wire_components()

        # Start WebSocket and REST API servers
        await self._websocket_server.start()
        await self._rest_api.start()
        
        # Initialize the controller
        await self._controller.initialize()
        
        self._is_running = True
        self._logger.system("servers_started", {})
    
    async def stop(self) -> None:
        """Stop all server components gracefully."""
        self._logger.system("servers_stopping", {})
        self._is_running = False
        
        await self._controller.shutdown()
        await self._websocket_server.stop()
        await self._rest_api.stop()
        
        self._logger.system("servers_stopped", {})
    
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
            self._logger.system("keyboard_interrupt", {})
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
        # Controller -> WebSocket (outbound) via domain events
        def handle_domain_event(event: DomainEvent) -> None:
            event_to_message_type = {
                DomainEventType.FEEDBACK_READY: MessageType.FEEDBACK_DELIVERY,
                DomainEventType.SYSTEM_STATUS_UPDATED: MessageType.STATUS_UPDATE,
                DomainEventType.EXPERIMENT_STARTED: MessageType.STATUS_UPDATE,
                DomainEventType.EXPERIMENT_ENDED: MessageType.STATUS_UPDATE,
            }

            message_type = event_to_message_type.get(event.event_type)
            if message_type is None:
                self._logger.system(
                    "unknown_domain_event_type",
                    {"event_type": getattr(event.event_type, "value", str(event.event_type))},
                    level="WARNING",
                )
                return

            recipient_id = (event.metadata or {}).get("recipient_id")

            msg = WebSocketMessage(
                type=message_type,
                timestamp=event.timestamp,
                payload=json_safe(event.payload),
                message_id=None,
                target_client_id=recipient_id,
            )

            def _handle_task_result(task: asyncio.Task) -> None:
                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    self._logger.system(
                        "background_task_cancelled",
                        {"source": "handle_domain_event"},
                        level="DEBUG",
                    )
                    return
                if exc is not None:
                    self._logger.system(
                        "background_task_error",
                        {
                            "source": "handle_domain_event",
                            "error": str(exc),
                        },
                        level="ERROR",
                    )
            if recipient_id:
                task = asyncio.create_task(self._websocket_server.send_to_client(recipient_id, msg))
            else:
                task = asyncio.create_task(self._broadcast_websocket_message(msg))
            task.add_done_callback(_handle_task_result)

        self._controller.register_event_handler(handle_domain_event)

        # WebSocket -> Controller (inbound)
        self._setup_websocket_handlers()

        # REST routes -> Controller (inbound)
        self._setup_api_routes()
    
    async def _broadcast_websocket_message(self, msg: WebSocketMessage) -> None:
        """Broadcast a WebSocket message to all connected clients."""
        try:
            await self._websocket_server.broadcast(msg)
            self._logger.system(
                "message_broadcast",
                {"message_type": msg.type.value},
                level="DEBUG",
            )
        except Exception as e:
            self._logger.system(
                "error_broadcasting_message",
                {"error": str(e)},
                level="ERROR",
            )
    
    def _setup_api_routes(self) -> None:
        """Set up REST API routes with controller handlers."""

        self._rest_api.register_route(
            "/status",
            HttpMethod.GET,
            self._controller.get_system_status,
        )

        self._rest_api.register_route(
            "/experiment/start",
            HttpMethod.POST,
            self._controller.start_experiment,
        )
        
        self._rest_api.register_route(
            "/experiment/end",
            HttpMethod.POST,
            self._controller.end_experiment,
        )

        self._rest_api.register_route(
            "/feedback/manual_send",
            HttpMethod.GET,
            self._controller.manual_send_feedback,
        )

        # TODO - add more routes as needed
    
    def _setup_websocket_handlers(self) -> None:
        """
        Set up WebSocket message handlers to handle messages correctly.
        """
        async def on_context_update(message: WebSocketMessage, client_id: str) -> None:
            self._logger.system(
                "context_update_received",
                {"client_id": client_id},
                level="DEBUG",
            )

            # Convert payload into your internal CodeContext type
            ctx = CodeContext.from_dict(message.payload)
            ctx.metadata = {**ctx.metadata, "requester_id": client_id}
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
        self._logger.system(
            "shutdown_signal",
            {"signal": sig.name},
        )
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
