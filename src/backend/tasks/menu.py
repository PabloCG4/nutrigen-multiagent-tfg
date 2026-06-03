from __future__ import annotations

import os
from typing import Any, Optional

from src.backend.redis_client import get_redis
from src.backend.state_store import update_menu_job_hash
from src.backend.celery_app import celery_app


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_int_optional(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


@celery_app.task(
    bind=True,
    name="menu.generate",
    soft_time_limit=int(os.getenv("MENU_TASK_SOFT_TIME_LIMIT") or 14 * 60),
    time_limit=int(os.getenv("MENU_TASK_TIME_LIMIT") or 15 * 60),
)
def generate_menu_task(  # noqa: ANN201
    self,
    job_id: str,
    user_id: int,
    payload: dict[str, Any],
    profile: dict[str, Any],
    output_language: str,
) -> dict[str, Any]:
    """
    CPU/IO-heavy LangGraph menu generation task.

    Important:
    - FastAPI (producer) enqueues this task by name via celery_app.send_task(...),
      so FastAPI does NOT need to import this module nor any src.agents.* code.
    - Heavy imports (LangGraph, LLM clients, RAG) are kept inside the task body.
    - Progress is reported directly to Redis via update_menu_job_hash, driven by
      graph node events (no time-based progress pumper threads).
    """
    if not isinstance(job_id, str) or job_id.strip() == "":
        raise ValueError("job_id is required")

    redis_client = get_redis()

    def progress_cb(state: str, progress: int, message: str) -> None:
        # Best-effort: never crash the task because Redis progress update failed.
        try:
            update_menu_job_hash(
                redis_client,
                job_id,
                status="running",
                progress=_safe_int(progress, 0),
                message=str(message),
            )
        except Exception:
            pass

    try:
        update_menu_job_hash(
            redis_client,
            job_id,
            status="running",
            progress=0,
            message="Running",
            error="",
        )

        # Local imports: keep producer process clean.
        from src.backend.utils import parse_dietary_style
        from src.agents.graph import run_pipeline
        from src.agents.mediator import UserContext

        parsed_dietary_style = parse_dietary_style(payload.get("dietary_style"))
        user_context = UserContext(
            available_minutes=_safe_int_optional(payload.get("time_available")),
            is_cheat_meal=False,
            special_requests=payload.get("special_requests"),
            output_language="es" if output_language not in ("es", "en") else output_language,
        )

        final_state = run_pipeline(
            barcodes=payload.get("barcodes"),
            diners=_safe_int(payload.get("diners"), 2),
            user_context=user_context,
            dietary_style=parsed_dietary_style,
            cheat_meal_description=None,
            dish_type_filter=payload.get("dish_type"),
            user_profile_data=profile,
            compute_phase3=True,
            progress_cb=progress_cb,
            priority_barcode=payload.get("priority_barcode"),
        )

        mediator_output = final_state.get("mediator_output")
        phase3_output = final_state.get("phase3_output")
        if mediator_output is None:
            raise RuntimeError("Pipeline completed without Mediator menu output.")
        if phase3_output is None:
            raise RuntimeError("Pipeline completed without Phase 3 output.")

        finalized_recipes: list[dict[str, Any]] = []
        phase3_macro_map: Optional[dict[str, Any]] = None
        for idx, recipe in enumerate(mediator_output.recipes):
            recipe_payload = recipe.model_dump()
            macro_breakdown = None
            if idx < len(phase3_output.recipes):
                macro_breakdown = phase3_output.recipes[idx].macro_breakdown
            else:
                if phase3_macro_map is None:
                    phase3_macro_map = {
                        r.recipe_title: r.macro_breakdown for r in phase3_output.recipes
                    }
                macro_breakdown = phase3_macro_map.get(recipe.title)
            if macro_breakdown is not None:
                recipe_payload["macro_breakdown"] = macro_breakdown.model_dump()
            finalized_recipes.append(recipe_payload)

        result_payload = {
            "recipes": finalized_recipes,
            "phase3_exact_output": phase3_output.model_dump(),
            "macros_pending": False,
        }

        update_menu_job_hash(
            redis_client,
            job_id,
            status="done",
            progress=100,
            message="Done",
            result=result_payload,
            error="",
        )
        return result_payload
    except Exception as exc:
        update_menu_job_hash(
            redis_client,
            job_id,
            status="error",
            progress=0,
            message="Error",
            error=str(exc),
        )
        raise

