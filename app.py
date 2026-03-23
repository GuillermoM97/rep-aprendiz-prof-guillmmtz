"""Compatibility entrypoint for Cloud Run.

The service's source of truth lives in ``main.py``.
This module simply re-exports the FastAPI app so existing tooling that
references ``app:app`` continues to work.
"""

from main import app
