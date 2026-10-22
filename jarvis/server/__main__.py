"""Run the Soundwave companion server: python -m jarvis.server"""

import uvicorn

from .. import config
from .app import create_app


def main() -> None:
    host = config.get("JARVIS_APP_HOST") or "0.0.0.0"
    port = config.get_int("JARVIS_APP_PORT") or 8080
    app = create_app()
    token = config.get("JARVIS_APP_TOKEN")
    if token:
        print(f"Soundwave app server on {host}:{port}")
        print(f"  token: {token}  (in .env as JARVIS_APP_TOKEN)")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
