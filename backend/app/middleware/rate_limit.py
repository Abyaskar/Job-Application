"""
Fixed-window rate limiter middleware.

Uses the same cache abstraction as recommendation-result caching (Redis in
production, in-memory dict in demo mode) so no separate infra is needed.
Fixed-window rather than sliding/token-bucket: simpler to reason about and
sufficient given this API's traffic profile (internal dashboard, not a
public high-QPS endpoint) -- documented as a scale trade-off in the README.
"""
from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.db.cache import get_cache


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if request.url.path in ("/health", "/health/live", "/health/ready", "/"):
            return await call_next(request)

        client_id = request.client.host if request.client else "unknown"
        window = int(time.time() // settings.RATE_LIMIT_WINDOW_SECONDS)
        key = f"ratelimit:{client_id}:{window}"

        cache = get_cache()
        count = await cache.incr(key)
        if count == 1:
            await cache.expire(key, settings.RATE_LIMIT_WINDOW_SECONDS)

        if count > settings.RATE_LIMIT_REQUESTS:
            return Response(
                content='{"detail":"Rate limit exceeded. Please slow down."}',
                status_code=429,
                media_type="application/json",
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(max(0, settings.RATE_LIMIT_REQUESTS - count))
        return response
