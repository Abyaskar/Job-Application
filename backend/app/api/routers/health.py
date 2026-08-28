from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.db.cache import get_cache
from app.db.mongo import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "app": get_settings().APP_NAME}


@router.get("/health/live")
async def liveness():
    """Process is up and serving traffic — used for k8s/Cloud Run liveness probes."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness():
    """Dependencies (DB/cache) are reachable — used for readiness probes so
    traffic isn't routed to an instance that can't yet serve requests."""
    checks = {"db": "unknown", "cache": "unknown"}
    try:
        get_db()
        checks["db"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["db"] = f"error: {exc}"
    try:
        get_cache()
        checks["cache"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["cache"] = f"error: {exc}"

    healthy = all(v == "ok" for v in checks.values())
    return {"status": "ready" if healthy else "degraded", "checks": checks}
