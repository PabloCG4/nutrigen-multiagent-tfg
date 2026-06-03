from __future__ import annotations

import os
from typing import Any

from src.backend.celery_app import celery_app


@celery_app.task(bind=True, name="vision.process_image")
def process_vision_image_task(self, image_path: str) -> dict[str, Any]:
    """
    Heavy vision inference task.

    Strict rule: Only this Celery worker process may import and execute the PyTorch/CUDA pipeline.
    FastAPI must never import src.vision.* to avoid duplicating VRAM allocations.
    """
    if not isinstance(image_path, str) or image_path.strip() == "":
        raise ValueError("image_path is required")

    normalized = image_path.strip()
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"Image not found: {normalized}")

    try:
        # Heavy import (PyTorch/CUDA) MUST live inside the worker process.
        from src.vision.inference.master_pipeline import extract_products_from_image

        products = extract_products_from_image(normalized)
        if not isinstance(products, list):
            products = []
        return {"products": products}
    finally:
        # Best-effort cleanup of temporary file.
        try:
            os.remove(normalized)
        except OSError:
            pass

