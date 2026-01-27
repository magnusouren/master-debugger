"""
REST API Server

Provides HTTP endpoints for configuration, status queries, and
non-real-time operations.

Design:
- Server.py decides *which* routes exist (wiring) by calling `register_route(...)`.
- RestAPI is a thin transport layer that binds registered routes into aiohttp.
- /health is kept as a built-in liveness endpoint.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Callable, Awaitable, Any
from enum import Enum
import json
import inspect
from backend.api.serialization import json_safe  
from aiohttp import web


class HttpMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


# Route handler takes a dict request envelope and returns a dict payload
RouteHandler = Callable[..., Awaitable[Any]] | Callable[..., Any]


class RestAPI:
    """
    REST API server for configuration and status endpoints.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
    ):
        self._host = host
        self._port = port
        self._app: Optional[object] = None  # aiohttp.web.Application
        self._runner: Optional[object] = None  # aiohttp.web.AppRunner
        self._routes: Dict[str, Dict[HttpMethod, RouteHandler]] = {}
        self._is_running: bool = False

    async def start(self) -> None:
        """Start the REST API server."""
        from aiohttp import web

        self._app = web.Application()
        self._add_middleware()
        self._setup_app()

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._is_running = True

    async def stop(self) -> None:
        """Stop the REST API server."""
        self._is_running = False
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._is_running

    def register_route(
        self,
        path: str,
        method: HttpMethod,
        handler: RouteHandler,
    ) -> None:
        """
        Register a route handler.

        Args:
            path: URL path (e.g., "/status")
            method: HttpMethod enum (e.g., HttpMethod.GET)
            handler: async function that takes request dict and returns response dict
        """
        if not path.startswith("/"):
            path = "/" + path

        if path not in self._routes:
            self._routes[path] = {}

        if method in self._routes[path]:
            raise ValueError(f"Route already registered: {method.value} {path}")

        self._routes[path][method] = handler

    # --- Internal Methods ---

    def _setup_app(self) -> None:
        """Bind built-in routes and all registered routes into aiohttp."""
        from aiohttp import web

        if self._app is None:
            raise RuntimeError("REST API app is not initialized")

        # Built-in liveness endpoint (does not depend on controller wiring)
        self._app.router.add_get("/health", self._health_handler)

        # Bind registered routes from server wiring
        for path, methods in self._routes.items():
            for method, handler in methods.items():
                aiohttp_handler = self._make_aiohttp_handler(handler)
                self._app.router.add_route(method.value, path, aiohttp_handler)

    def _add_middleware(self) -> None:
        """Add middleware for logging, CORS, etc. (optional)."""
        if self._app is None:
            return

        # Keep minimal for now; you can expand later.
        # Example place to add CORS headers or request logging middleware.
        # (No-op by default.)
        return

    def _make_aiohttp_handler(self, handler):

        async def _wrapped(request: "web.Request") -> "web.Response":
            try:
                req = await self._request_to_dict(request)

                sig = inspect.signature(handler)
                params = sig.parameters

                # Prepare arguments
                call_result = None

                if len(params) == 0:
                    call_result = handler()

                elif len(params) == 1:
                    # One param: pass full request dict
                    call_result = handler(req)

                else:
                    # Multiple params: map from JSON body or query
                    payload = req.get("json") or {}
                    if not isinstance(payload, dict):
                        raise ValueError("Expected JSON object body")

                    # Only pass expected parameters
                    kwargs = {
                        name: payload[name]
                        for name in params.keys()
                        if name in payload
                    }

                    missing = [
                        name for name in params.keys()
                        if name not in kwargs
                    ]
                    if missing:
                        raise ValueError(f"Missing required fields: {missing}")

                    call_result = handler(**kwargs)

                if inspect.isawaitable(call_result):
                    call_result = await call_result

                call_result = json_safe(call_result)

                if call_result is None:
                    call_result = {"status": "ok"}

                return self._create_response(call_result, status=200)

            except web.HTTPException:
                raise

            except Exception as e:
                return self._create_error_response(f"Internal server error: {e}", status=500)

        return _wrapped

    async def _request_to_dict(self, request: object) -> Dict[str, Any]:
        """
        Convert aiohttp Request into a simple dict envelope.
        Includes:
          - method, path
          - query params
          - headers (subset)
          - json body (if present), otherwise raw text (if any)
        """
        # We keep typing loose to avoid importing aiohttp types at module import time.
        method = getattr(request, "method", None)
        path = getattr(request, "path", None)

        # query params
        query = {}
        rel_url = getattr(request, "rel_url", None)
        if rel_url is not None:
            q = getattr(rel_url, "query", None)
            if q is not None:
                query = dict(q)

        # headers (as dict)
        headers = {}
        hdrs = getattr(request, "headers", None)
        if hdrs is not None:
            headers = dict(hdrs)

        # body: try json, fallback to text
        body_json: Any = None
        body_text: Optional[str] = None

        try:
            # aiohttp request has .json()
            body_json = await request.json()
        except Exception:
            try:
                body_text = await request.text()
                if body_text == "":
                    body_text = None
            except Exception:
                body_text = None

        return {
            "method": method,
            "path": path,
            "query": query,
            "headers": headers,
            "json": body_json,
            "text": body_text,
        }

    def _create_response(
        self,
        data: Dict[str, Any],
        status: int = 200,
    ) -> object:
        """Create an HTTP JSON response."""
        from aiohttp import web

        # Ensure it's JSON-serializable (basic)
        try:
            json.dumps(data)
        except TypeError:
            data = {"error": "Response not JSON serializable"}

        return web.json_response(data, status=status)

    def _create_error_response(
        self,
        message: str,
        status: int = 400,
    ) -> object:
        """Create an error response."""
        return self._create_response({"error": message}, status=status)

    # --- Built-in Handlers ---

    async def _health_handler(self, request) -> "object":
        """Health check endpoint handler."""
        from aiohttp import web

        return web.json_response({"status": "ok"})