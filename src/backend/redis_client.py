from __future__ import annotations

import os
from typing import Optional

import redis

_pool: Optional[redis.ConnectionPool] = None


def get_redis_url() -> str:
    return (os.getenv("REDIS_URL") or "").strip() or "redis://localhost:6379/0"


def get_redis_pool() -> redis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            get_redis_url(),
            decode_responses=True,
        )
    return _pool


def get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=get_redis_pool())
