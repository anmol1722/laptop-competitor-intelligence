"""Web layer for the laptop competitor-intelligence system.

``api.py`` exposes the backend modules (pipeline / matching / pricing / sentiment /
rag) over JSON and serves the static UI from ``src/web/static``.

Run it with::

    .venv/bin/python -m uvicorn src.web.api:app --port 8000
    # or
    .venv/bin/python src/web/api.py --port 8000
"""

__all__ = ["api"]
