from __future__ import annotations

"""
Redis-backed ephemeral state for horizontal scalability.

Keyspace:
- OAuth CSRF state:   oauth_state:{state}    (string value, TTL-managed)
- Menu generation job: menu_job:{job_id}     (Redis Hash, TTL-managed)

TTL policy (call site constants):
- OAuth state: 600s (matches _OAUTH_STATE_TTL_SECONDS)
- Menu jobs:   2h   (matches _MENU_JOB_TTL_SECONDS)
"""

import json
from typing import Any, Optional

import redis


def oauth_state_key(state: str) -> str:
    return f"oauth_state:{state}"


def menu_job_key(job_id: str) -> str:
    return f"menu_job:{job_id}"


def set_oauth_state(redis_client: redis.Redis, state: str, *, ttl_seconds: int) -> None:
    # Value is not used; existence + TTL is what matters.
    redis_client.set(oauth_state_key(state), "1", ex=int(ttl_seconds))


def consume_oauth_state(redis_client: redis.Redis, state: str) -> bool:
    """
    Returns True if the state existed and was consumed (deleted), False otherwise.
    """
    key = oauth_state_key(state)
    if redis_client.exists(key):
        redis_client.delete(key)
        return True
    return False


def create_menu_job_hash(
    redis_client: redis.Redis,
    *,
    job_id: str,
    user_id: int,
    created_at: float,
    status: str,
    progress: int,
    message: str,
    ttl_seconds: int,
) -> None:
    key = menu_job_key(job_id)
    redis_client.hset(
        key,
        mapping={
            "job_id": str(job_id),
            "user_id": str(int(user_id)),
            "created_at": str(float(created_at)),
            "status": str(status),
            "progress": str(int(progress)),
            "message": str(message),
            "result": "",
            "error": "",
        },
    )
    redis_client.expire(key, int(ttl_seconds))


def update_menu_job_hash(
    redis_client: redis.Redis,
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    result: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    key = menu_job_key(job_id)
    mapping: dict[str, str] = {}
    if status is not None:
        mapping["status"] = str(status)
    if progress is not None:
        mapping["progress"] = str(int(progress))
    if message is not None:
        mapping["message"] = str(message)
    if result is not None:
        mapping["result"] = json.dumps(result, ensure_ascii=False)
    if error is not None:
        mapping["error"] = str(error)
    if mapping:
        redis_client.hset(key, mapping=mapping)


def get_menu_job_hash(redis_client: redis.Redis, job_id: str) -> Optional[dict[str, Any]]:
    key = menu_job_key(job_id)
    raw = redis_client.hgetall(key)
    if not raw:
        return None
    result_raw = (raw.get("result") or "").strip()
    error_raw = (raw.get("error") or "").strip()
    parsed_result: Any = None
    if result_raw != "":
        try:
            parsed_result = json.loads(result_raw)
        except json.JSONDecodeError:
            parsed_result = None

    def parse_int(v: Any, default: int) -> int:
        try:
            return int(float(v))
        except Exception:
            return default

    def parse_float(v: Any, default: float) -> float:
        try:
            return float(v)
        except Exception:
            return default

    return {
        "job_id": raw.get("job_id") or job_id,
        "user_id": parse_int(raw.get("user_id"), 0),
        "created_at": parse_float(raw.get("created_at"), 0.0),
        "status": raw.get("status") or "",
        "progress": parse_int(raw.get("progress"), 0),
        "message": raw.get("message") or "",
        "result": parsed_result,
        "error": error_raw if error_raw != "" else None,
    }

