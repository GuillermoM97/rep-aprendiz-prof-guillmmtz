"""Cloud Run-friendly structured logging utilities.

This module mirrors the template's logging behavior, but with a safe module
name for imports from the service entrypoint.
"""

from __future__ import annotations

import json
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs
    structlog = None

_TRACE_HEADER = "X-Cloud-Trace-Context"
_request_headers: ContextVar[Optional[Mapping[str, str]]] = ContextVar(
    "request_headers", default=None
)



def bind_request_headers(headers: Optional[Mapping[str, str]]) -> None:
    """Bind request headers for the current request context."""
    _request_headers.set(headers)



def clear_request_headers() -> None:
    """Clear request headers from the current request context."""
    _request_headers.set(None)



def _project_trace() -> Optional[str]:
    headers = _request_headers.get()
    if not headers:
        return None

    trace_header = headers.get(_TRACE_HEADER)
    if not trace_header:
        return None

    try:
        import metadata

        project = metadata.get_project_id()
    except Exception:
        return None

    trace = trace_header.split("/")
    return f"projects/{project}/traces/{trace[0]}"


class _FallbackLogger:
    """Very small JSON logger used when structlog is unavailable."""

    def _emit(self, severity: str, event: str, **kwargs: Any) -> None:
        payload: Dict[str, Any] = {
            "severity": severity,
            "message": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        trace = _project_trace()
        if trace:
            payload["logging.googleapis.com/trace"] = trace
        payload.update(kwargs)
        print(json.dumps(payload, default=str))

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit("ERROR", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        kwargs.setdefault("traceback", traceback.format_exc())
        self._emit("ERROR", event, **kwargs)



def field_name_modifier(
    logger: "structlog.PrintLogger", log_method: str, event_dict: Dict
) -> Dict:
    """Change key names to match Cloud Logging expectations."""
    event_dict["severity"] = event_dict["level"].upper()
    del event_dict["level"]

    if "event" in event_dict:
        event_dict["message"] = event_dict["event"]
        del event_dict["event"]
    return event_dict



def trace_modifier(
    logger: "structlog.PrintLogger", log_method: str, event_dict: Dict
) -> Dict:
    """Add Cloud Trace correlation when request headers are available."""
    trace = _project_trace()
    if trace:
        event_dict["logging.googleapis.com/trace"] = trace
    return event_dict



def get_json_logger():
    """Create a JSON logger for Cloud Run / Cloud Logging."""
    if structlog is None:
        return _FallbackLogger()

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            field_name_modifier,
            trace_modifier,
            structlog.processors.TimeStamper("iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    return structlog.get_logger()


logger = get_json_logger()



def flush() -> None:
    """No-op because stdout is already unbuffered in the container."""
    pass
