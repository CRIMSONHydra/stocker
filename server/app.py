"""Server entry point and ASGI application export.

The actual application lives in app/main.py. This module preserves
`server.app:app` as an ASGI target and adds a callable `main()`
entry point required by OpenEnv validation.
"""

from app.main import app, run  # noqa: F401

__all__ = ["app", "main"]


def main() -> None:
    """CLI/server entry point required by OpenEnv validation."""
    run()


if __name__ == "__main__":
    main()
