"""
Cache + rate-limit abstraction.

Redis is used for three distinct jobs in this system, deliberately kept
behind one small interface:
  1. Result caching -- ranking a candidate against thousands of jobs is
     expensive (embedding similarity + rule evaluation for every job), so
     identical (candidate_version, filter) queries are cached with a TTL.
  2. Rate limiting -- a fixed-window counter per API key/IP protects the
     (comparatively expensive) recommendation endpoint from abuse.
  3. De-duplication of repeated searches -- search query hashes are cached
     so paginating or re-filtering the same result set doesn't recompute.

In DEMO_MODE we swap in an in-process dict with manual TTL bookkeeping. It
is not a general Redis emulator -- only GET/SET/EXPIRE/INCR semantics used
by this app are implemented.
"""
from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("db.cache")


class InMemoryCache:
    def __init__(self):
        self._store: dict[str, tuple[Any, float | None]] = {}
        self.hits = 0
        self.misses = 0

    def _expired(self, expires_at: float | None) -> bool:
        return expires_at is not None and time.time() > expires_at

    async def get(self, key: str) -> str | None:
        item = self._store.get(key)
        if not item or self._expired(item[1]):
            self.misses += 1
            return None
        self.hits += 1
        return item[0]

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        expires_at = time.time() + ex if ex else None
        self._store[key] = (value, expires_at)

    async def incr(self, key: str) -> int:
        item = self._store.get(key)
        current = int(item[0]) if item and not self._expired(item[1]) else 0
        current += 1
        expires_at = item[1] if item and not self._expired(item[1]) else None
        self._store[key] = (str(current), expires_at)
        return current

    async def expire(self, key: str, seconds: int) -> None:
        item = self._store.get(key)
        if item:
            self._store[key] = (item[0], time.time() + seconds)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class RedisCache:
    def __init__(self, client: aioredis.Redis):
        self._client = client
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> str | None:
        val = await self._client.get(key)
        if val is None:
            self.misses += 1
            return None
        self.hits += 1
        return val

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        await self._client.set(key, value, ex=ex)

    async def incr(self, key: str) -> int:
        return await self._client.incr(key)

    async def expire(self, key: str, seconds: int) -> None:
        await self._client.expire(key, seconds)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


_cache = None


async def connect_to_redis():
    global _cache
    settings = get_settings()
    if settings.DEMO_MODE:
        logger.info("redis.demo_mode", detail="using in-process cache")
        _cache = InMemoryCache()
        return
    client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    await client.ping()
    _cache = RedisCache(client)
    logger.info("redis.connected", url=settings.REDIS_URL)


def get_cache() -> InMemoryCache | RedisCache:
    if _cache is None:
        raise RuntimeError("Cache not initialized. Call connect_to_redis() first.")
    return _cache


async def cache_get_json(key: str) -> Any | None:
    raw = await get_cache().get(key)
    return json.loads(raw) if raw else None


async def cache_set_json(key: str, value: Any, ttl: int | None = None) -> None:
    ttl = ttl or get_settings().CACHE_TTL_SECONDS
    await get_cache().set(key, json.dumps(value), ex=ttl)
