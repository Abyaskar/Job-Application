from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.memory_store import InMemoryDatabase

logger = get_logger("db.mongo")

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | InMemoryDatabase | None = None


async def connect_to_mongo() -> None:
    global _client, _db
    settings = get_settings()
    if settings.DEMO_MODE:
        logger.info("mongo.demo_mode", detail="using in-memory Mongo-compatible store")
        _db = InMemoryDatabase()
        return
    _client = AsyncIOMotorClient(settings.MONGO_URI, uuidRepresentation="standard")
    _db = _client[settings.MONGO_DB_NAME]
    await _db.jobs.create_index("job_id", unique=True)
    await _db.candidates.create_index("candidate_id", unique=True)
    await _db.recommendations.create_index([("candidate_id", 1), ("job_id", 1)])
    logger.info("mongo.connected", uri=settings.MONGO_URI, db=settings.MONGO_DB_NAME)


async def close_mongo_connection() -> None:
    if _client:
        _client.close()
        logger.info("mongo.closed")


def get_db():
    if _db is None:
        raise RuntimeError("Mongo connection not initialized. Call connect_to_mongo() first.")
    return _db
