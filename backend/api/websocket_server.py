"""
WebSocket Server

Handles real-time bidirectional communication with the VS Code extension.
Used for streaming eye-tracking data, sending feedback, and receiving
context updates.
"""
from typing import Optional, Dict, Any, Set, Callable, Awaitable
import asyncio
import json
from dataclasses import asdict

from backend.types.messages import FeedbackMessage, MessageType, SystemStatusMessage, WebSocketMessage



# Type alias for message handlers
MessageHandler = Callable[[WebSocketMessage, str], Awaitable[None]]


class WebSocketServer:
    """
    WebSocket server for VS Code extension communication.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
    ):
        """
        Initialize the WebSocket server.
        
        Args:
            host: Host to bind to.
            port: Port to listen on.
        """
        self._host = host
        self._port = port
        self._server: Optional[object] = None  # websockets.WebSocketServer
        self._clients: Set[object] = set()  # Set of connected clients
        self._client_info: Dict[str, Dict[str, Any]] = {}  # Client metadata
        self._message_handlers: Dict[MessageType, MessageHandler] = {}
        self._is_running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def start(self) -> None:
        """Start the WebSocket server."""
        import websockets
        
        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port
        )
        self._is_running = True
        self._loop = asyncio.get_event_loop()


    async def stop(self) -> None:
        """Stop the WebSocket server and disconnect all clients."""
        self._is_running = False
        
        # Close all client connections
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        
        # Stop the server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
    
    def is_running(self) -> bool:
        """
        Check if server is running.
        
        Returns:
            True if server is running.
        """
        pass  # TODO: Implement status check
    
    def get_connected_clients(self) -> int:
        """
        Get number of connected clients.
        
        Returns:
            Number of connected clients.
        """
        pass  # TODO: Implement client count
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: MessageHandler,
    ) -> None:
        """
        Register a handler for a message type.
        
        Args:
            message_type: Type of message to handle.
            handler: Async function to handle the message.
        """

        self._message_handlers[message_type] = handler

    
    def unregister_handler(self, message_type: MessageType) -> None:
        """
        Unregister a handler for a message type.
        
        Args:
            message_type: Type of message to unregister.
        """

        if message_type in self._message_handlers:
            del self._message_handlers[message_type]
    
    async def send_to_client(
        self,
        client_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """
        Send a message to a specific client.
        
        Args:
            client_id: ID of the target client.
            message: Message to send.
            
        Returns:
            True if sent successfully.
        """
        pass  # TODO: Implement targeted sending
    
    async def broadcast(self, message: WebSocketMessage) -> int:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast.
            
        Returns:
            Number of clients message was sent to.
        """
        pass  # TODO: Implement broadcasting
    
    async def send_feedback(self, feedback: FeedbackMessage) -> bool:
        """
        Send feedback to VS Code extension.
        
        Args:
            feedback: Feedback message to send.
            
        Returns:
            True if sent successfully.
        """
        pass  # TODO: Implement feedback sending
    
    async def send_status_update(self, status: SystemStatusMessage) -> bool:
        """
        Send status update to connected clients.
        
        Args:
            status: Status message to send.
            
        Returns:
            True if sent successfully.
        """
        pass  # TODO: Implement status update sending
    
    async def request_context(self, client_id: Optional[str] = None) -> None:
        """
        Request code context from VS Code.
        
        Args:
            client_id: Optional specific client to request from.
        """
        pass  # TODO: Implement context request
    
    # --- Internal Methods ---
    
    async def _handle_connection(self, websocket: object, path: str = "") -> None:
        """
        Handle a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection.
            path: Connection path.
        """
        import uuid
        client_id = str(uuid.uuid4())
        self._clients.add(websocket)
        self._client_info[client_id] = {
            "websocket": websocket,
            "connected_at": asyncio.get_event_loop().time(),
        }
        
        print(f"  Client connected: {client_id}")
        
        try:
            async for message in websocket:
                await self._process_message(message, client_id)
        except Exception as e:
            print(f"  Client error: {e}")
        finally:
            await self._handle_disconnection(client_id)
    
    async def _handle_disconnection(self, client_id: str) -> None:
        """
        Handle client disconnection.
        
        Args:
            client_id: ID of disconnected client.
        """
        print(f"  Client disconnected: {client_id}")
        if client_id in self._client_info:
            websocket = self._client_info[client_id].get("websocket")
            if websocket in self._clients:
                self._clients.discard(websocket)
            del self._client_info[client_id]
    
    async def _receive_messages(
        self, 
        websocket: object, 
        client_id: str
    ) -> None:
        """
        Receive and process messages from a client.
        
        Args:
            websocket: The WebSocket connection.
            client_id: ID of the client.
        """
        async for message in websocket:
            await self._process_message(message, client_id)
    
    async def _process_message(
        self, 
        raw_message: str, 
        client_id: str
    ) -> None:
        """
        Process a received message.
        
        Args:
            raw_message: Raw JSON message string.
            client_id: ID of the sending client.
        """
        message = self._parse_message(raw_message)
        if message:
            # Log context updates
            if message.type == MessageType.CONTEXT_UPDATE:
                # payload: ContextUpdate = message.payload  # Explicitly typing payload as ContextUpdate
                print(f"  [WebSocket] Processing context update from client {client_id}: ")

            handler = self._message_handlers.get(message.type)
            if handler:
                try:
                    await handler(message, client_id)
                except Exception as e:
                    print(
                        f"[WebSocket] Handler error for {message.type}: "
                        f"{type(e).__name__}: {e}"
                    )
                
    def _parse_message(self, raw_message: str) -> Optional[WebSocketMessage]:
        """
        Parse a raw message into a WebSocketMessage.
        
        Args:
            raw_message: Raw JSON message string.
            
        Returns:
            Parsed message or None if invalid.
        """
        import json
        try:
            data = json.loads(raw_message)
            return WebSocketMessage(
                type=MessageType(data.get("type")),
                timestamp=data.get("timestamp", 0),
                payload=data.get("payload", {}),
                message_id=data.get("messageId"),
            )
        except (json.JSONDecodeError, ValueError):
            return None
    
    def _serialize_message(self, message: WebSocketMessage) -> str:
        """
        Serialize a message to JSON string.
        
        Args:
            message: Message to serialize.
            
        Returns:
            JSON string.
        """
        pass  # TODO: Implement message serialization
    
    def _generate_client_id(self) -> str:
        """
        Generate a unique client ID.
        
        Returns:
            Unique client ID string.
        """
        pass  # TODO: Implement ID generation
    
    async def _ping_clients(self) -> None:
        """Send periodic ping to keep connections alive."""
        pass  # TODO: Implement ping logic
