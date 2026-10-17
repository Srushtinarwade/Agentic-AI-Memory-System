import os
import redis

# Connect to Redis using the configured URL.
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0"
)

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True
)


def check_redis_connection() -> bool:
    """Check whether Redis is reachable."""
    return bool(redis_client.ping())