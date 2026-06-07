import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .config import get_settings

logger = logging.getLogger("webapp.db")

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_db() -> None:
    global _client, _db
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongo_uri)
    _db = _client[settings.mongo_db]
    # Indexes
    await _db.users.create_index("email", unique=True)
    await _db.users.create_index("username")
    await _db.threads.create_index("user_id")
    await _db.messages.create_index("thread_id")
    await _db.messages.create_index([("thread_id", 1), ("created_at", 1)])
    logger.info("MongoDB connected: %s / %s", settings.mongo_uri, settings.mongo_db)


async def close_db() -> None:
    global _client
    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialised. Call connect_db() first.")
    return _db