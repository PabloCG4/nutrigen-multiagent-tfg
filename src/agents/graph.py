"""
src/agents/graph.py
────────────────────────────────────────────────────────────────────────────────
End-to-end orchestrator that connects the Computer Vision barcode extraction
module with the multi-agent LangGraph culinary pipeline.

Pipeline flow
─────────────
  [Vision loop]  →  [Context collection]  →  [LangGraph]
                                                   │
                          ┌────────────────────────┘
                          ▼
                    ── INITIAL ROUTING ──
                   /                     \
          is_cheat_meal?              NOT cheat meal
                  │                        │
                  ▼                        ▼
     Node 0: Cheat-Meal Tracker   Node 1: Nutritionist Phase 1
     (macro estimate, terminal)   (allergen filter + baseline)
                  │                        │
                 END               Node 2: Chef Agent
                                   (retrieval + adaptation)
                                           │
                                   Node 3: Nutritionist Phase 2
                                   (macro audit + corrections)
                                           │
                                   Node 4: Mediator Agent
                                   (culinary integration, 3 recipes)
                                           │
                                   Node 5: Nutritionist Phase 3
                                   (exact macro calculation)
                                           │
                                          END

Cheat-meal note
───────────────
When the user declares a cheat meal, they describe it in free text.
The graph routes directly to the Cheat-Meal Tracker (Nutritionist) which
estimates macros from the description and terminates — no Chef or Mediator.

In the normal path, Phase 2 always runs (no bypass), and Phase 3 computes
exact per-serving macros for all 3 final recipes using OpenFoodFacts data.
"""
from __future__ import annotations

import contextvars
import os
import textwrap
import time
from pathlib import Path
from typing import Callable, List, Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.agents import config
from src.agents.chef import ChefAgent, ChefProposal, ChefRequest
from src.agents.mediator import (
    FinalIngredient,
    FinalRecipe,
    IngredientSource,
    MediatorAgent,
    MediatorOutput,
    MediatorRequest,
    UserContext,
)
from src.agents.nutritionist import (
    ActivityLevel,
    CheatMealReport,
    DietaryStyle,
    Gender,
    MacroBreakdown,
    NutritionistAgent,
    NutritionistCritique,
    NutritionistReport,
    Phase3Output,
    PhysicalGoal,
    UserProfile,
)
from src.agents.runtime import get_agents as _get_agents

ProgressCallback = Callable[[str, int, str], None]
_progress_cb_var: contextvars.ContextVar[Optional[ProgressCallback]] = contextvars.ContextVar(
    "pipeline_progress_cb",
    default=None,
)


def _emit_progress(state: str, progress: int, message: str) -> None:
    cb = _progress_cb_var.get()
    if cb is None:
        return
    try:
        cb(str(state), int(progress), str(message))
    except Exception:
        # Progress reporting must never break the graph execution.
        return


# ─────────────────────────────────────────────────────────────────────────────
# DEMO USER PROFILE TEMPLATE
# Barcodes are injected at runtime from the vision module.
# TODO: replace with a real profile form (age, weight, allergies, macros…)
#       once the user-onboarding module is implemented.
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_PROFILE: dict = dict(
    age=30,
    gender=Gender.MALE,
    weight_kg=80.0,
    height_cm=180.0,
    activity_level=ActivityLevel.MODERATELY_ACTIVE,
    physical_goal=PhysicalGoal.MAINTENANCE,
    meals_per_day=3,
    protein_pct=25,
    carb_pct=50,
    fat_pct=25,
    allergies_and_intolerances=[],
)


# Agent singletons are centralized in src/agents/runtime.py.


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """Shared mutable state passed between graph nodes."""
    barcodes:               List[str]
    diners:                 int
    user_context:           UserContext
    compute_phase3:         bool
    user_profile_data:      Optional[dict]
    dietary_style:          Optional[DietaryStyle]
    dish_type_filter:       Optional[str]              # ChromaDB dish_type pre-filter
    priority_barcode:       Optional[str]              # user-pinned ingredient for RAG queries
    cheat_meal_description: Optional[str]              # populated only on the cheat-meal path
    cheat_meal_report:      Optional[CheatMealReport]  # output of Node 0
    nutritionist_report:    Optional[NutritionistReport]
    chef_proposal:          Optional[ChefProposal]
    nutritionist_critique:  Optional[NutritionistCritique]
    mediator_output:        Optional[MediatorOutput]
    phase3_output:          Optional[Phase3Output]     # output of Node 5


class PipelineProgressStep(TypedDict):
    state: str
    progress: int


def get_pipeline_progress_plan(is_cheat_meal: bool, compute_phase3: bool = True) -> List[PipelineProgressStep]:
    """
    Returns a progress plan used by streaming endpoints.

    Progress is intentionally coarse-grained: it keeps the HTTP connection alive
    during multi-agent inference without depending on internal LangGraph events.
    """
    if is_cheat_meal:
        return [
            {"state": "initializing", "progress": 10},
            {"state": "cheat_meal_processing", "progress": 70},
            {"state": "finalizing", "progress": 95},
        ]

    base = [
        {"state": "initializing", "progress": 5},
        {"state": "nutritionist_phase1", "progress": 35},
        {"state": "chef_phase2", "progress": 60},
        {"state": "mediator_phase3", "progress": 80},
    ]
    if compute_phase3:
        base.append({"state": "finalizing", "progress": 95})
    else:
        base.append({"state": "finalizing", "progress": 90})
    return base


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH NODES
# Each node receives the full PipelineState and returns a dict with only the
# keys it wants to update; LangGraph merges the result into the current state.
# ─────────────────────────────────────────────────────────────────────────────

def _node_cheat_meal_tracker(state: PipelineState) -> dict:
    """
    Node 0 (cheat-meal path only): uses the Nutritionist to estimate macros
    for the user's free-text meal description.  This is the terminal node on
    the cheat-meal branch — the pipeline ends here.
    """
    print("\n" + "─" * 56)
    print("[Graph] NODE 0 — Cheat-Meal Tracker (Nutritionist)")
    print("─" * 56)
    _emit_progress("cheat_meal_processing", 70, "Cheat meal processing")

    nutritionist, _, _ = _get_agents()
    description = state.get("cheat_meal_description") or "indulgent meal (no description)"
    t0 = time.perf_counter()
    report = nutritionist.estimate_cheat_meal(description)
    dt = time.perf_counter() - t0
    print(f"[Graph][Timing] Node0 cheat_meal_tracker: {dt:.2f}s")
    return {"cheat_meal_report": report}


def _node_nutritionist_phase1(state: PipelineState) -> dict:
    print("\n" + "─" * 56)
    print("[Graph] NODE 1 — Nutritionist: Phase 1 (Baseline)")
    print("─" * 56)
    _emit_progress("nutritionist_phase1", 35, "Nutritionist phase 1")

    nutritionist, _, _ = _get_agents()
    
    db_data = state.get("user_profile_data")
    
    if db_data:
        # 1. Mathematical reconstruction of percentages from the DB grams
        kcal = db_data.get("target_calories") or 2000.0
        p_pct = int(round((db_data.get("target_protein_g", 0) * 4 / kcal) * 100))
        c_pct = int(round((db_data.get("target_carb_g", 0) * 4 / kcal) * 100))
        f_pct = int(round((db_data.get("target_fat_g", 0) * 9 / kcal) * 100))
        
        # 2. Correction of rounding to satisfy the Pydantic validator (must be 100 exact)
        total_pct = p_pct + c_pct + f_pct
        if total_pct != 100 and total_pct > 0:
            c_pct += (100 - total_pct)

        # 3. DB keys -> Pydantic UserProfile
        profile_kwargs = {
            "age": db_data.get("age"),
            "gender": db_data.get("gender"),
            "weight_kg": db_data.get("weight"),
            "height_cm": db_data.get("height"),
            "activity_level": db_data.get("activity_level"),
            "physical_goal": db_data.get("physical_goal"),
            "meals_per_day": int(db_data.get("meals_per_day") or 3),  # Valor por defecto estático
            "protein_pct": p_pct,
            "carb_pct": c_pct,
            "fat_pct": f_pct,
            "target_calories": db_data.get("target_calories"),
            "target_protein_g": db_data.get("target_protein_g"),
            "target_carb_g": db_data.get("target_carb_g"),
            "target_fat_g": db_data.get("target_fat_g"),
            # Canonical EU-14 English ids (see src/backend/allergy_ids.py); migrated on profile read.
            "allergies_and_intolerances": db_data.get("allergies", []),
            # Propagate adaptive session parameters into the user profile contract
            "session_meal_caloric_target": db_data.get("session_meal_caloric_target"),
            "session_meal_protein_g": db_data.get("session_meal_protein_g"),
            "session_meal_carb_g": db_data.get("session_meal_carb_g"),
            "session_meal_fat_g": db_data.get("session_meal_fat_g"),
        }
    else:
        profile_kwargs = dict(_DEMO_PROFILE)

    # 4. Session context
    if state.get("dietary_style") is not None:
        profile_kwargs["dietary_style"] = state["dietary_style"]

    profile = UserProfile(**profile_kwargs, barcodes=state["barcodes"])
    t0 = time.perf_counter()
    report = nutritionist.run(profile)
    dt = time.perf_counter() - t0
    print(f"[Graph][Timing] Node1 nutritionist_phase1: {dt:.2f}s")
    return {"nutritionist_report": report}


def _node_chef(state: PipelineState) -> dict:
    print("\n" + "─" * 56)
    print("[Graph] NODE 2 — Chef Agent")
    print("─" * 56)
    _emit_progress("chef_phase2", 55, "Chef phase 2")

    nutritionist, chef, _ = _get_agents()
    ctx: UserContext = state["user_context"]
    request = ChefRequest(
        nutritionist_report=state["nutritionist_report"],
        diners=state["diners"],
        dietary_style=state.get("dietary_style"),
        available_minutes=ctx.available_minutes,
        dish_type_filter=state.get("dish_type_filter"),
        output_language=ctx.output_language,
        priority_barcode=state.get("priority_barcode"),
    )
    t0 = time.perf_counter()
    proposal = chef.run(request)
    proposal = nutritionist.hydrate_chef_calorie_estimates(
        report=state["nutritionist_report"],
        proposal=proposal,
        diners=state["diners"],
    )
    dt = time.perf_counter() - t0
    print(f"[Graph][Timing] Node2 chef: {dt:.2f}s")
    return {"chef_proposal": proposal}


def _node_nutritionist_phase2(state: PipelineState) -> dict:
    print("\n" + "─" * 56)
    print("[Graph] NODE 3 — Nutritionist: Phase 2 (Macro Audit)")
    print("─" * 56)
    _emit_progress("nutritionist_phase2", 65, "Nutritionist phase 2")

    nutritionist, _, _ = _get_agents()
    t0 = time.perf_counter()
    critique = nutritionist.evaluate_proposals(
        state["nutritionist_report"],
        state["chef_proposal"],
        diners=state["diners"],
        output_language=state["user_context"].output_language,
    )
    dt = time.perf_counter() - t0
    print(f"[Graph][Timing] Node3 nutritionist_phase2: {dt:.2f}s")
    return {"nutritionist_critique": critique}


def _node_mediator(state: PipelineState) -> dict:
    print("\n" + "─" * 56)
    print("[Graph] NODE 4 — Mediator Agent")
    print("─" * 56)
    _emit_progress("mediator_phase3", 80, "Mediator phase 3")

    _, _, mediator = _get_agents()

    # Phase 2 always runs on the normal path — nutritionist_critique is guaranteed
    critique: NutritionistCritique = state["nutritionist_critique"]

    request = MediatorRequest(
        user_context=state["user_context"],
        chef_proposal=state["chef_proposal"],
        nutritionist_critique=critique,
        diners=state["diners"],
    )
    t0 = time.perf_counter()
    output = mediator.run(request)
    dt = time.perf_counter() - t0
    print(f"[Graph][Timing] Node4 mediator: {dt:.2f}s")
    return {"mediator_output": output}

def _route_after_mediator(state: PipelineState) -> str:
    """Optionally skips Phase 3 when running in fast mode."""
    return "nutritionist_phase3" if state.get("compute_phase3", True) else "end_fast"


def _node_nutritionist_phase3(state: PipelineState) -> dict:
    """
    Node 5: Nutritionist Phase 3 — macro calculation for the 3 final recipes.

    Computes per-serving macros deterministically:
      - Uses stored per-100g profiles from Phase 1 for ingredients matched to the scan.
      - Estimates per-100g macros only for ingredients not covered by the stored profiles.
    """
    print("\n" + "─" * 56)
    print("[Graph] NODE 5 — Nutritionist: Phase 3 (Macro Calculation)")
    print("─" * 56)
    _emit_progress("finalizing", 95, "Finalizing")

    nutritionist, _, _ = _get_agents()
    t0 = time.perf_counter()
    adjusted_mediator = nutritionist.adjust_mediator_recipes_to_meal_nutrition_targets(
        mediator_output=state["mediator_output"],
        nutritionist_report=state["nutritionist_report"],
        diners=state["diners"],
    )
    t_adj = time.perf_counter()
    print(f"[Graph][Timing] Node5a portion_adjust: {t_adj - t0:.2f}s")
    phase3 = nutritionist.calculate_final_macros(
        mediator_output=adjusted_mediator,
        nutritionist_report=state["nutritionist_report"],
        diners=state["diners"],
    )
    dt = time.perf_counter() - t_adj
    print(f"[Graph][Timing] Node5 nutritionist_phase3: {dt:.2f}s")
    return {
        "phase3_output": phase3,
        "mediator_output": adjusted_mediator,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def _route_initial(state: PipelineState) -> str:
    """Routes to the cheat-meal tracker or the full nutritional pipeline."""
    if state["user_context"].is_cheat_meal:
        print("[Graph] Cheat-meal mode → routing to Cheat-Meal Tracker.")
        return "cheat_meal_tracker"
    return "nutritionist_phase1"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH COMPILATION
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Assembles and compiles the LangGraph StateGraph.

    Topology:
        START
          ├─[cheat_meal]────► cheat_meal_tracker ──────────────────────────► END
          └─[normal]────────► nutritionist_phase1
                                    └─► chef
                                          └─► nutritionist_phase2
                                                    └─► mediator
                                                              └─► nutritionist_phase3 ──► END
    """
    graph = StateGraph(PipelineState)

    # Register all nodes
    graph.add_node("cheat_meal_tracker",    _node_cheat_meal_tracker)
    graph.add_node("nutritionist_phase1",   _node_nutritionist_phase1)
    graph.add_node("chef",                  _node_chef)
    graph.add_node("nutritionist_phase2",   _node_nutritionist_phase2)
    graph.add_node("mediator",              _node_mediator)
    graph.add_node("nutritionist_phase3",   _node_nutritionist_phase3)
    graph.add_node("end_fast",              lambda _state: {})

    # Initial conditional routing from START
    graph.add_conditional_edges(
        START,
        _route_initial,
        {
            "cheat_meal_tracker":  "cheat_meal_tracker",
            "nutritionist_phase1": "nutritionist_phase1",
        },
    )

    # Cheat-meal path terminates immediately after tracking
    graph.add_edge("cheat_meal_tracker", END)

    # Normal path: linear flow (Phase 2 always runs)
    graph.add_edge("nutritionist_phase1", "chef")
    graph.add_edge("chef",                "nutritionist_phase2")
    graph.add_edge("nutritionist_phase2", "mediator")
    graph.add_conditional_edges(
        "mediator",
        _route_after_mediator,
        {
            "nutritionist_phase3": "nutritionist_phase3",
            "end_fast": "end_fast",
        },
    )
    graph.add_edge("nutritionist_phase3", END)
    graph.add_edge("end_fast", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# VISION MODULE — IMAGE INGESTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _extract_barcodes_from_products(final_products: list) -> List[str]:
    """
    Extracts the top-1 barcode from each product dict returned by the vision pipeline.
    Deduplicates while preserving detection order.
    """
    seen: set[str] = set()
    barcodes: List[str] = []
    for product in final_products:
        if product.get("top_matches"):
            barcode: str = product["top_matches"][0]["barcode"]
            if barcode not in seen:
                seen.add(barcode)
                barcodes.append(barcode)
    return barcodes


def run_vision_ingestion_loop() -> List[str]:
    """
    Interactive console loop that accepts image paths, runs the vision pipeline
    on each one, and accumulates the deduplicated barcodes across all images.

    If the vision module cannot be imported (missing GPU dependencies, model
    weights not found, etc.), falls back to manual barcode entry so the rest
    of the pipeline can still be tested.

    Returns:
        Deduplicated list of EAN/UPC barcodes detected across all images.
    """
    all_barcodes: List[str] = []

    print("\n" + "=" * 60)
    print("  VISION MODULE — Barcode Extraction")
    print("=" * 60)
    print("Enter full image paths one by one (fridge / pantry photos).")
    print("Type 'done' or press Enter on an empty line to continue.\n")

    # Lazy import — vision deps (ultralytics, SAM, EasyOCR) are heavy and
    # may not be installed in environments that only run the agent pipeline.
    vision_fn = None
    try:
        from src.vision.inference.master_pipeline import extract_products_from_image
        vision_fn = extract_products_from_image
        print("[Vision] Module loaded successfully.\n")
    except Exception as e:
        print(f"[WARNING] Vision module unavailable ({e}).")
        print("[WARNING] Falling back to manual barcode entry.\n")

    while True:
        raw = input("  Image path (or 'done'): ").strip()
        if raw.lower() in ("done", "exit", "quit", ""):
            break

        # Strip surrounding quotes that Windows Explorer may add on drag-and-drop
        image_path = raw.strip('"').strip("'")

        if not os.path.exists(image_path):
            print(f"  [!] File not found: {image_path}\n")
            continue

        if vision_fn is not None:
            try:
                products = vision_fn(image_path)
                new_barcodes = _extract_barcodes_from_products(products)
                print(f"\n  [Vision] {len(new_barcodes)} unique barcode(s) detected:")
                for bc in new_barcodes:
                    if bc not in all_barcodes:
                        all_barcodes.append(bc)
                        print(f"    [+] {bc}")
                    else:
                        print(f"    [~] {bc}  (already registered)")
                print()
            except Exception as e:
                print(f"  [!] Vision error on '{image_path}': {e}\n")
        else:
            # Manual fallback: user types barcodes associated with this image
            print(f"  Manual mode for: {image_path}")
            print("  Enter barcodes one per line; empty line to finish.\n")
            while True:
                bc = input("    Barcode: ").strip()
                if not bc:
                    break
                if bc not in all_barcodes:
                    all_barcodes.append(bc)
                    print(f"      [+] {bc} registered")
            print()

    if not all_barcodes:
        print(
            "[WARNING] No barcodes collected.\n"
            "[WARNING] Using demo barcodes (Nutella + wheat flour) for testing.\n"
        )
        all_barcodes = [
            "3017620422003",  # Nutella  (contains milk → substitute_required)
            "3155250349793",  # Wheat flour → approved
        ]

    return all_barcodes


# ─────────────────────────────────────────────────────────────────────────────
# USER CONTEXT COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_int(prompt: str, default: int, min_val: int, max_val: int) -> int:
    """Prompts for an integer within [min_val, max_val]; returns default on empty."""
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"    [!] Must be between {min_val} and {max_val}.")
        except ValueError:
            print("    [!] Please enter a valid integer.")


def _prompt_optional_int(prompt: str, min_val: int, max_val: int) -> Optional[int]:
    """Prompts for an optional integer; returns None on empty input."""
    while True:
        raw = input(f"  {prompt} [Enter to skip]: ").strip()
        if not raw:
            return None
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"    [!] Must be between {min_val} and {max_val}.")
        except ValueError:
            print("    [!] Please enter a valid integer or press Enter to skip.")


# Maps user text input to a DietaryStyle enum value
_DIETARY_ALIASES: dict[str, DietaryStyle] = {
    "vegan":          DietaryStyle.VEGAN,
    "vegano":         DietaryStyle.VEGAN,
    "vegana":         DietaryStyle.VEGAN,
    "vegetarian":     DietaryStyle.VEGETARIAN,
    "vegetariano":    DietaryStyle.VEGETARIAN,
    "vegetariana":    DietaryStyle.VEGETARIAN,
    "celiac":         DietaryStyle.CELIAC,
    "celiaco":        DietaryStyle.CELIAC,
    "celiaca":        DietaryStyle.CELIAC,
    "mediterranean":  DietaryStyle.MEDITERRANEAN,
    "mediterranea":   DietaryStyle.MEDITERRANEAN,
    "mediterráneo":   DietaryStyle.MEDITERRANEAN,
    "mediterránea":   DietaryStyle.MEDITERRANEAN,
    "asian":          DietaryStyle.ASIAN,
    "asiatica":       DietaryStyle.ASIAN,
    "asiática":       DietaryStyle.ASIAN,
    "arabic":         DietaryStyle.ARABIC,
    "arabe":          DietaryStyle.ARABIC,
    "árabe":          DietaryStyle.ARABIC,
    "latin":          DietaryStyle.LATIN,
    "latina":         DietaryStyle.LATIN,
    "latino":         DietaryStyle.LATIN,
}


_DISH_TYPE_VALUES: frozenset[str] = frozenset({
    "soup_stew", "salad_side", "main_course",
    "dessert", "snack_breakfast", "drink",
})

def collect_session_context() -> tuple[int, UserContext, Optional[DietaryStyle], Optional[str], Optional[str]]:
    """
    Collects the dynamic session parameters interactively from the console.

    Returns:
        (diners, UserContext, dietary_style, cheat_meal_description, dish_type_filter)
        - dietary_style is None when the user skips (profile default is used).
        - cheat_meal_description is a non-empty string only when is_cheat_meal=True.
        - dish_type_filter is a valid dish_type string or None when the user skips.
    """
    print("\n" + "=" * 60)
    print("  SESSION CONTEXT")
    print("=" * 60)

    # ── Cheat-meal gate (asked FIRST — determines the entire remainder of the flow) ──
    cheat_raw = input("  Cheat meal? (yes/no) [no]: ").strip().lower()
    is_cheat_meal = cheat_raw in ("yes", "y", "s", "sí", "si")

    cheat_meal_description: Optional[str] = None
    if is_cheat_meal:
        print("  [Cheat meal mode] The nutritionist will track calories for your meal.")
        desc_raw = input(
            "  What are you eating? (e.g. 'double cheeseburger, large fries, Coke'): "
        ).strip()
        cheat_meal_description = desc_raw or "an indulgent meal (no details provided)"
        # Diners, cooking time, dietary preference, and special requests are all
        # irrelevant for cheat-meal tracking — skip them entirely.
        ctx = UserContext(is_cheat_meal=True)
        return 1, ctx, None, cheat_meal_description, None

    # ── Normal flow — collect remaining context ──
    diners = _prompt_int(
        "Number of diners", default=2, min_val=1, max_val=20
    )

    available_minutes = _prompt_optional_int(
        "Available cooking time in minutes (5–240)", min_val=5, max_val=240
    )

    diet_raw = input(
        "  Dietary preference? "
        "(vegan/vegetarian/celiac/mediterranean/asian/arabic/latin — Enter to skip): "
    ).strip().lower()
    dietary_style: Optional[DietaryStyle] = _DIETARY_ALIASES.get(diet_raw)
    if diet_raw and dietary_style is None:
        print(f"    [!] '{diet_raw}' not recognised — dietary preference skipped.")

    dish_raw = input(
        "  What type of dish? "
        "(soup_stew / salad_side / main_course / dessert / snack_breakfast / drink"
        " — Enter to skip): "
    ).strip().lower()
    dish_type_filter: Optional[str] = dish_raw if dish_raw in _DISH_TYPE_VALUES else None
    if dish_raw and dish_type_filter is None:
        print(f"    [!] '{dish_raw}' not recognised — dish type filter skipped.")

    special_raw = input(
        "  Special requests (e.g. 'something spicy', press Enter to skip): "
    ).strip()
    special_requests: Optional[str] = special_raw or None

    ctx = UserContext(
        available_minutes=available_minutes,
        is_cheat_meal=False,
        special_requests=special_requests,
    )
    return diners, ctx, dietary_style, None, dish_type_filter


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

_W = 66  # Console width for the output box

_SOURCE_TAG = {
    IngredientSource.AVAILABLE:             "[OK] ",
    IngredientSource.TO_BUY:                "[BUY]",
    IngredientSource.NUTRITIONIST_ADDITION: "[+]  ",
}
_SOURCE_LABEL = {
    IngredientSource.AVAILABLE:             "tienes en casa",
    IngredientSource.TO_BUY:                "necesitas comprar",
    IngredientSource.NUTRITIONIST_ADDITION: "sugerido por el nutricionista",
}


def _print_single_recipe(recipe: FinalRecipe, index: Optional[int] = None) -> None:
    """
    Renders one FinalRecipe in a structured, human-readable console format.
    If `index` is supplied it is shown as "OPCIÓN N" in the header.
    """
    bar  = "=" * _W
    thin = "-" * _W

    def section(title: str) -> None:
        print(thin)
        print(f"  {title}")
        print(thin)

    def wrap(text: str, indent: int = 4, width: int = _W - 4) -> List[str]:
        return textwrap.wrap(text, width=width, initial_indent=" " * indent,
                             subsequent_indent=" " * indent)

    header = f"OPCIÓN {index}" if index is not None else "RECETA"
    print("\n" + bar)
    print(f"  {header}: {recipe.title}")
    print(bar)
    print(f"  Tiempo    : {recipe.estimated_time_minutes} minutos")
    print(f"  Calorias  : {recipe.caloric_note}")
    print()

    section("JUSTIFICACIÓN")
    for line in wrap(recipe.justification):
        print(line)

    section("INGREDIENTES")
    groups = [
        (IngredientSource.AVAILABLE,             "Tienes en casa:"),
        (IngredientSource.TO_BUY,                "Necesitas comprar:"),
        (IngredientSource.NUTRITIONIST_ADDITION, "Sugeridos por el nutricionista:"),
    ]
    for source, label in groups:
        items = [i for i in recipe.ingredients if i.source == source]
        if not items:
            continue
        print(f"\n  {label}")
        for ing in items:
            tag = _SOURCE_TAG[source]
            print(f"    {tag} {ing.name:<38} {ing.quantity_g:>5} g")

    section("PASOS DE PREPARACIÓN")
    for idx, step in enumerate(recipe.preparation_steps, start=1):
        prefix = f"  {idx:>2}. "
        continuation = "      "
        lines = textwrap.wrap(step, width=_W - len(prefix))
        print(prefix + lines[0])
        for line in lines[1:]:
            print(continuation + line)

    print(bar)


def print_cheat_meal_report(report: CheatMealReport) -> None:
    """Renders the cheat-meal macro estimate in a compact console format."""
    bar  = "=" * _W
    thin = "-" * _W
    print("\n" + bar)
    print("  SEGUIMIENTO DE CHEAT MEAL")
    print(bar)
    print(f"  Comida    : {report.meal_description}")
    print()
    print(thin)
    print("  ESTIMACIÓN NUTRICIONAL")
    print(thin)
    print(f"  Calorías     : {report.estimated_kcal:>6} kcal")
    print(f"  Proteínas    : {report.estimated_protein_g:>6.1f} g")
    print(f"  Carbohidratos: {report.estimated_carb_g:>6.1f} g")
    print(f"  Grasas       : {report.estimated_fat_g:>6.1f} g")
    print()
    print(thin)
    print("  NOTA")
    print(thin)
    wrap_lines = textwrap.wrap(report.estimation_notes, width=_W - 4)
    for line in wrap_lines:
        print(f"  {line}")
    print(bar + "\n")


def print_final_menu(
    output: MediatorOutput,
    phase3: Optional[Phase3Output] = None,
) -> None:
    """
    Renders all 3 FinalRecipe options in a structured, human-readable console format.
    When `phase3` is provided, appends exact macro totals beneath each recipe.
    """
    bar = "=" * _W
    print("\n" + bar)
    print("  MENÚ FINAL — 3 OPCIONES")
    print(f"  Elige la receta que más te apetezca preparar hoy.")
    print(bar)

    # Build a quick lookup: title → MacroBreakdown
    macro_map: dict[str, MacroBreakdown] = {}
    if phase3:
        for rn in phase3.recipes:
            macro_map[rn.recipe_title] = rn.macro_breakdown

    for i, recipe in enumerate(output.recipes, start=1):
        _print_single_recipe(recipe, index=i)

        # Append Phase 3 macro breakdown if available
        mb = macro_map.get(recipe.title)
        if mb:
            thin = "-" * _W
            print(thin)
            print("  ANÁLISIS NUTRICIONAL EXACTO (por ración)")
            print(thin)
            print(f"    Calorías     : {mb.total_kcal:>6} kcal")
            print(f"    Proteínas    : {mb.protein_g:>6.1f} g")
            print(f"    Carbohidratos: {mb.carb_g:>6.1f} g")
            print(f"    Grasas       : {mb.fat_g:>6.1f} g")
            note_lines = textwrap.wrap(mb.accuracy_note, width=_W - 6)
            for line in note_lines:
                print(f"    {line}")
            print("=" * _W)
        print()

    print(bar + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    barcodes: Optional[List[str]] = None,
    diners: int = 2,
    user_context: Optional[UserContext] = None,
    dietary_style: Optional[DietaryStyle] = None,
    cheat_meal_description: Optional[str] = None,
    dish_type_filter: Optional[str] = None,
    user_profile_data: Optional[dict] = None,
    compute_phase3: bool = True,
    progress_cb: Optional[ProgressCallback] = None,
    priority_barcode: Optional[str] = None,
) -> PipelineState:
    """
    Executes the full multi-agent pipeline and returns the final graph state.

    The graph handles two distinct paths automatically:
      • Cheat-meal path  (user_context.is_cheat_meal=True):
            state["cheat_meal_report"] is populated; all recipe fields are None.
      • Normal path:
            state["mediator_output"] and state["phase3_output"] are populated.

    Args:
        barcodes:               EAN/UPC barcodes from vision; defaults to demo codes.
        diners:                 Number of people to cook for.
        user_context:           Session context (time, cheat-meal, requests).
        dietary_style:          Optional dietary preference (overrides profile default).
        cheat_meal_description: Free-text meal description (only for cheat-meal path).

    Returns:
        PipelineState — the final LangGraph state; check populated fields to determine
        which path was taken.
    """
    if barcodes is None:
        barcodes = ["3017620422003", "3155250349793"]
    if user_context is None:
        user_context = UserContext()

    app = build_graph()

    initial_state: PipelineState = {
        "barcodes":               barcodes,
        "diners":                 diners,
        "user_context":           user_context,
        "compute_phase3":         compute_phase3,
        "user_profile_data":      user_profile_data,
        "dietary_style":          dietary_style,
        "dish_type_filter":       dish_type_filter,
        "priority_barcode":       priority_barcode,
        "cheat_meal_description": cheat_meal_description,
        "cheat_meal_report":      None,
        "nutritionist_report":    None,
        "chef_proposal":          None,
        "nutritionist_critique":  None,
        "mediator_output":        None,
        "phase3_output":          None,
    }

    print("\n" + "=" * 60)
    print("  CULINARY AI — Multi-agent Pipeline")
    print(f"  Barcodes  : {len(barcodes)} product(s)")
    print(f"  Diners    : {diners}")
    cheat_flag = "YES" if user_context.is_cheat_meal else "NO"
    print(f"  Cheat meal: {cheat_flag}")
    if not user_context.is_cheat_meal:
        diet_flag = dietary_style.value if dietary_style else "none"
        print(f"  Diet      : {diet_flag}")
        dish_flag = dish_type_filter or "any"
        print(f"  Dish type : {dish_flag}")
    print("=" * 60)
    token = _progress_cb_var.set(progress_cb)
    try:
        _emit_progress("initializing", 5, "Initializing")
        final_state: PipelineState = app.invoke(initial_state)
        return final_state
    finally:
        _progress_cb_var.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CULINARY AI ASSISTANT")
    print("  End-to-end Pipeline: Vision → Agents → 3 Recipes")
    print("=" * 60)

    # Phase 0 — Vision ingestion
    barcodes = run_vision_ingestion_loop()
    print(f"\n[Graph] Collected {len(barcodes)} unique barcode(s): {barcodes}")

    # Phase 0b — Session context (returns 5-tuple)
    diners, user_context, dietary_style, cheat_meal_description, dish_type_filter = (
        collect_session_context()
    )

    # Phase 1–5 (or 0 for cheat meal) — LangGraph pipeline
    try:
        final_state = run_pipeline(
            barcodes=barcodes,
            diners=diners,
            user_context=user_context,
            dietary_style=dietary_style,
            cheat_meal_description=cheat_meal_description,
            dish_type_filter=dish_type_filter,
        )
    except Exception as pipeline_error:
        print(f"\n[CRITICAL] Pipeline failed: {pipeline_error}")
        raise

    # Display results — branch on which path was taken
    if final_state.get("cheat_meal_report"):
        print_cheat_meal_report(final_state["cheat_meal_report"])
    else:
        print_final_menu(
            final_state["mediator_output"],
            phase3=final_state.get("phase3_output"),
        )
