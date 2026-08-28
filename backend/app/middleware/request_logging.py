from __future__ import annotations

import time

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger, new_request_id

logger = get_logger("http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = new_request_id()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("http.request.failed")
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info("http.request", status_code=response.status_code, duration_ms=duration_ms)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = str(duration_ms)
        return response
