# Backend package
# Import Server lazily to avoid circular imports

__version__ = "0.1.0"


def get_server():
    """Get the Server class (lazy import)."""
    from backend.api.server import Server
    return Server


def create_server(config_path=None):
    """Factory function to create a server instance."""
    from backend.api.server import create_server as _create_server
    return _create_server(config_path)


__all__ = [
    "get_server",
    "create_server",
]
