from __future__ import annotations
import os
import ssl
from pathlib import Path
from dotenv import load_dotenv

# Ensure env vars are available even when imported before FastAPI loads .env
load_dotenv(Path(__file__).resolve().parent / ".env")
from celery import Celery


def _redis_url() -> str:
    return (os.getenv("REDIS_URL") or "").strip() or "redis://localhost:6379/0"


def _celery_broker_url() -> str:
    return (os.getenv("CELERY_BROKER_URL") or "").strip() or _redis_url()


def _celery_result_backend() -> str:
    return (os.getenv("CELERY_RESULT_BACKEND") or "").strip() or _redis_url()


def _ssl_options_for_url(url: str) -> dict:
    """
    Robust SSL options for Upstash (rediss://) broker/backend.
    Uses CERT_REQUIRED + CA bundle when available.
    """
    if not isinstance(url, str) or not url.lower().startswith("rediss://"):
        return {}

    ca_certs = None
    try:
        import certifi  # type: ignore

        ca_certs = certifi.where()
    except Exception:
        ca_certs = None

    options: dict = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }
    if ca_certs:
        options["ssl_ca_certs"] = ca_certs
    return options


celery_app = Celery(
    "tfg",
    broker=_celery_broker_url(),
    backend=_celery_result_backend(),
    include=["src.backend.tasks.vision", "src.backend.tasks.menu"],
)

_BROKER_URL = _celery_broker_url()
_BACKEND_URL = _celery_result_backend()
_BROKER_SSL = _ssl_options_for_url(_BROKER_URL)
_BACKEND_SSL = _ssl_options_for_url(_BACKEND_URL)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_time_limit=int(os.getenv("VISION_TASK_TIME_LIMIT") or 180),
    task_soft_time_limit=int(os.getenv("VISION_TASK_SOFT_TIME_LIMIT") or 150),
    broker_use_ssl=_BROKER_SSL or None,
    redis_backend_use_ssl=_BACKEND_SSL or None,
)

