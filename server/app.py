"""
server/app.py
=============
OpenEnv multi-mode deployment entry point.
[project.scripts] server = "server.app:main"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401


def main() -> None:
    """Entry point called by 'server' CLI script and OpenEnv runner."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()