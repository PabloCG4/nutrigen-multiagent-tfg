from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Type

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load the project .env before any LangChain / OpenAI client is instantiated,
# so OPENAI_API_KEY is available from the moment the module is imported.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

from src.agents.tools.openfoodfacts_tool import OpenFoodFactsTool
from src.agents import config
from src.common.product_data import get_product_macros_mapping
from src.common.nutrition_targets import round_half_up

if TYPE_CHECKING:
    from src.agents.mediator import MediatorOutput

_local_macros_cache: Optional[dict[str, dict]] = None


def _get_local_macros_mapping() -> dict[str, dict]:
    """
    Loads barcode -> { product_name, kcal_per_100g, protein_g_per_100g, ... } from local JSON.
    Cached after first read to avoid repeated disk IO during agent runs.
    """
    global _local_macros_cache
    if _local_macros_cache is not None:
        return _local_macros_cache

    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    _local_macros_cache = get_product_macros_mapping(data_dir)
    return _local_macros_cache


def _fmt_local_profile(barcode: str, record: dict) -> str:

    """
    Formats a local product profile for LLM consumption.
    Returns a string with the barcode, product name, and macronutrients.
    """
    name = record.get("product_name") if isinstance(record.get("product_name"), str) else "Unknown Product"
    kcal = record.get("kcal_per_100g")
    p = record.get("protein_g_per_100g")
    c = record.get("carbs_g_per_100g")
    f = record.get("fat_g_per_100g")

    def _num_or_na(v) -> str:
        if isinstance(v, (int, float)):
            return str(round(float(v), 2))
        return "N/A"

    return (
        f"[LOCAL] {barcode} | {name}\n"
        f"kcal/100g: {_num_or_na(kcal)} | P: {_num_or_na(p)}g | C: {_num_or_na(c)}g | F: {_num_or_na(f)}g\n"
        f"Allergens: Unknown"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Nutritionist Agent in a multi-agent culinary assistant system.

Your SOLE responsibility is to establish the nutritional baseline for the current session.
You must NOT search for or suggest recipes — that is the Chef agent's job in the next pipeline node.

You will receive:
  1. A structured user health profile.
  2. Pre-calculated nutrition targets, computed externally via Mifflin-St Jeor + NEAT baseline:
       • DAILY CALORIC TARGET — total daily caloric budget (kcal).
       • MEAL CALORIC TARGET  — per-meal caloric budget (kcal).
       • DAILY MACRO TARGETS  — protein, carbohydrates, and fat in grams.
  3. A list of NUTRITIONAL PROFILES, one per detected product, fetched from OpenFoodFacts.

Complete the following step.

────────────────────────────────────────────
STEP 1 — Ingredient Assessment
────────────────────────────────────────────
For EVERY product in the nutritional profiles, produce one IngredientAssessment entry:

  a-DIET) DIETARY STYLE EXCLUSION — check this FIRST, before any allergen rule:
     Apply ONLY when `dietary_style` in the user profile is VEGAN, VEGETARIAN, or CELIAC.
     Skip entirely if `dietary_style` is null or a cultural type (mediterranean, asian, etc.).

       • VEGAN: Exclude if the product IS or CONTAINS any of: meat (beef, pork, lamb, chicken,
           turkey, duck, game), seafood, fish, shellfish, dairy (milk, butter, cream, cheese,
           yogurt, whey), eggs, honey, gelatin, or any other clearly animal-derived ingredient.
           → status = "excluded", reason = "Contains animal product — violates vegan diet."

       • VEGETARIAN: Exclude if the product IS or CONTAINS meat, poultry, fish, seafood, or
           shellfish. Dairy and eggs are permitted under vegetarian diet.
           → status = "excluded", reason = "Contains meat/fish — violates vegetarian diet."

       • CELIAC: Exclude if the product contains or is derived from wheat, rye, barley, spelt,
           kamut, triticale, or any gluten-containing grain. Also exclude on "may contain
           gluten/wheat" cross-contamination warnings — celiac requires zero-tolerance.
           → status = "excluded", reason = "Contains gluten — celiac requires strict exclusion."

     When a-DIET exclusion fires, skip rules b), c), d) for this product.
     Continue to sub-steps f) CALORIC DENSITY and g) CULINARY ROLE regardless.

  a) Cross-reference the product's allergen list with the user's `allergies_and_intolerances`.
     (Only applies when a-DIET did NOT fire for this product.)
     IMPORTANT SCOPING RULE: only act on the specific items listed in `allergies_and_intolerances`.
     Do NOT infer additional restrictions that the user has not declared. A product that contains
     milk, tree nuts, or soy is perfectly safe unless the user has explicitly listed those as restrictions.

     User-declared allergens use canonical English tokens from the EU-14 set, e.g. gluten, milk,
     peanuts, tree_nuts, crustaceans, molluscs, fish, eggs, soy, celery, mustard, sesame, sulfites, lupin.

  b) SEVERE ALLERGY (e.g., peanuts, tree_nuts, crustaceans, molluscs, wheat for celiac disease, fish):
       This applies ONLY when the user has explicitly declared the allergen AND the product's
       ingredient list shows it as a primary/main ingredient (not merely a trace or "may contain" warning).
       status = "excluded"
       reason = concise explanation naming the allergen and the matching declared restriction
       substitution_suggestion = null

  c) INTOLERANCE (e.g., lactose intolerance, non-celiac gluten sensitivity):
       This applies ONLY when the user has explicitly declared the intolerance AND the product
       contains that substance as a MAIN ingredient (e.g., wheat flour for gluten intolerance).
       "May contain traces of X" warnings do NOT trigger this rule for intolerances — traces are
       generally safe for non-celiac / non-allergic intolerances.
       status = "substitute_required"
       reason = concise explanation of the intolerance conflict
       substitution_suggestion = a concrete, practical alternative drawn from your internal knowledge
                                 (e.g., "lactose-free milk", "gluten-free oat flour")

  d) SAFE for the user (DEFAULT — use this unless a clear conflict is identified above):
       status = "approved"
       reason = brief confirmation that no declared restriction is violated
       substitution_suggestion = null

  e-ERROR) If the nutritional profile for a barcode contains a [TOOL ERROR] or [NOT FOUND] prefix,
       the product is UNKNOWN — the absence of data is NOT evidence of an allergen conflict.
       status = "approved"
       reason = "Product not found in OpenFoodFacts; approved by default — no allergen conflict can be confirmed."
       substitution_suggestion = null
       kcal_per_100g = estimated_value (see rule f below — estimate when missing)

  f) CALORIC DENSITY: For EVERY ingredient, regardless of its status, locate the
     "Calories" line in its OpenFoodFacts nutritional profile:
       Format in the profile text: "Calories       : X.XX kcal"
     Read that numeric value and assign it exactly to the `kcal_per_100g` field.
     If the value reads "N/A" or the profile contains [TOOL ERROR] / [NOT FOUND]:
       - Estimate kcal_per_100g for 100 g using your internal nutritional knowledge,
         based on the Product Name and Category in the profile text.
       - Use a realistic typical value for that food type (avoid extreme outliers).
       - ONLY set kcal_per_100g = 0.0 if it is truly impossible to infer any plausible food type.
 
  f2) MACROGRAMS PER 100G (protein/carbs/fat):
     From the same nutritional profile, locate the following lines:
       "Proteins       : X.XX g"
       "Carbohydrates  : X.XX g"
       "Fats           : X.XX g"
     Read those numeric values and assign them exactly to:
       protein_g_per_100g, carbs_g_per_100g, fat_g_per_100g.
     If any value reads "N/A" or the profile contains [TOOL ERROR] / [NOT FOUND]:
       - Estimate ONLY the missing macro(s) per 100 g using your internal nutritional knowledge
         based on the Product Name and Category.
       - Keep estimates realistic and internally consistent with kcal_per_100g.

  g) CULINARY ROLE: Assign exactly one role to EVERY ingredient using your food knowledge:
       • "main_protein"   — primary protein source: meat, poultry, fish, eggs, legumes, tofu.
       • "main_carb"      — primary carbohydrate source: rice, pasta, bread, potato, oats, flour.
       • "produce"        — fresh or canned vegetables and fruits.
       • "base_condiment" — cooking fats, spices, herbs, sauces, seasonings, vinegar, stock.
       • "snack_drink"    — packaged snacks, confectionery, spreads, beverages, desserts.
     When in doubt, choose the role that best describes the ingredient's PRIMARY function
     in a typical cooked meal (e.g., eggs = "main_protein" even though they are versatile).
     If the profile contains [TOOL ERROR] or [NOT FOUND], use "base_condiment" as the default.

  h) GENERIC NAMES — *** THIS FIELD IS MANDATORY FOR EVERY INGREDIENT. YOU MUST NEVER OMIT IT. ***
     Extract the clean culinary name for this ingredient in BOTH English and Spanish, stripping
     ALL brand names, pack sizes, quality grades, origins, and commercial adjectives.
     These names are passed DIRECTLY to the recipe search engine as query keywords.
     A wrong or missing value here causes the entire recipe-search pipeline to fail.

     *** RULE: generic_name_en MUST be a real food word in ENGLISH (never "unknown ingredient"
     unless the product is genuinely unrecognisable after reading its name and profile). ***
     *** RULE: generic_name_es MUST be a real food word in SPANISH. ***
     *** RULE: If the Product Name is a commercial brand or unclear, strictly use the Category field provided in the nutritional profile to deduce the generic_name_en and generic_name_es. ***

     Examples of correct extraction:
       product_name = "Queso fresco de burgos oveja KENT"
         → generic_name_en = "fresh cheese"    generic_name_es = "queso fresco"
       product_name = "MANGO kent premium"
         → generic_name_en = "mango"           generic_name_es = "mango"
       product_name = "Copos de avena suaves integrales"
         → generic_name_en = "oats"            generic_name_es = "avena"
       product_name = "Aceite de oliva virgen extra"
         → generic_name_en = "olive oil"       generic_name_es = "aceite de oliva"
       product_name = "Huevos frescos camperos M"
         → generic_name_en = "eggs"            generic_name_es = "huevos"

     Rules:
       — 1 to 2 words maximum; prefer a single precise noun or "adjective noun" pair.
       — generic_name_en: ENGLISH only — no Spanish words whatsoever.
       — generic_name_es: SPANISH only — no English words whatsoever.
       — For [TOOL ERROR] / [NOT FOUND]: use your food knowledge to infer from `product_name`;
         only fall back to "unknown ingredient" / "ingrediente desconocido" if truly impossible.

────────────────────────────────────────────
OUTPUT CONTEXT — Base Constraints for the Chef Agent
────────────────────────────────────────────
Your structured output will be passed as "Base Constraints" to the Chef Agent (the next pipeline
node). The Chef will use your report to propose recipes that:
  • Stay within the meal_caloric_target per dish (the hard per-meal ceiling).
  • Respect the daily_caloric_target as the overall daily budget.
  • Hit the macro targets: daily_protein_g, daily_carb_g, daily_fat_g.
  • Only use ingredients with status "approved" or find safe replacements for "substitute_required".
  • Avoid all "excluded" ingredients entirely.
  • The `culinary_role` of each ingredient guides the Chef's RAG query strategy:
    main proteins and carbs become primary search terms; base condiments and snack_drinks
    are excluded from queries (they don't define a dish's identity).
  • The `generic_name_en` and `generic_name_es` are clean culinary terms used DIRECTLY as RAG search
    keywords — they replace verbose commercial product names to avoid semantic dilution.

CRITICAL: All pre-calculated values (`daily_caloric_target`, `meal_caloric_target`,
`daily_protein_g`, `daily_carb_g`, `daily_fat_g`) in your JSON output MUST contain exactly the
integer values provided to you in the context. Do not recalculate or modify them.
"""

_CHEAT_MEAL_SYSTEM_PROMPT = """You are the Nutritionist Agent performing cheat meal tracking.

The user has decided to enjoy an indulgent meal and described it in free text.
Your task: estimate the nutritional content of that meal for a single person.

Use realistic, standard restaurant or home portion sizes when quantities are not specified.
If the description is vague (e.g., "a burger"), assume a typical medium-sized portion.

OUTPUT RULES:
  • `estimated_kcal`    — integer, total kilocalories for the meal as described.
  • `estimated_protein_g`, `estimated_carb_g`, `estimated_fat_g` — floats in grams.
  • `estimation_notes`  — 1–2 sentences noting which parts were estimated and confidence level.
"""


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class DietaryStyle(str, Enum):
    # Exclusion-based: mark non-compliant scanned ingredients as excluded
    VEGAN        = "vegan"
    VEGETARIAN   = "vegetarian"
    CELIAC       = "celiac"
    # Inclusion-based / cultural: steers the Chef's recipe-search queries only
    MEDITERRANEAN = "mediterranean"
    ASIAN         = "asian"
    ARABIC        = "arabic"
    LATIN         = "latin"


class ActivityLevel(str, Enum):
    SEDENTARY          = "sedentary"
    LIGHTLY_ACTIVE     = "lightly_active"
    MODERATELY_ACTIVE  = "moderately_active"
    VERY_ACTIVE        = "very_active"
    EXTREMELY_ACTIVE   = "extremely_active"


class PhysicalGoal(str, Enum):
    WEIGHT_LOSS  = "weight_loss"
    MAINTENANCE  = "maintenance"
    MUSCLE_GAIN  = "muscle_gain"


class Gender(str, Enum):
    MALE   = "male"
    FEMALE = "female"


class UserProfile(BaseModel):
    age: int = Field(..., ge=1, le=120, description="User's age in years")
    gender: Gender = Field(..., description="Biological sex used for BMR calculation")
    weight_kg: float = Field(..., gt=0, description="User's weight in kilograms")
    height_cm: float = Field(..., gt=0, description="User's height in centimetres")
    activity_level: ActivityLevel = Field(
        ...,
        description="Lifestyle / occupational activity tier (NEAT). Excludes deliberate exercise.",
    )
    physical_goal: PhysicalGoal = Field(..., description="Desired health or body composition goal")
    meals_per_day: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Number of meals per day; used to divide the daily target into per-meal budgets"
    )
    protein_pct: int = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of daily calories allocated to protein (e.g. 30)"
    )
    carb_pct: int = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of daily calories allocated to carbohydrates (e.g. 45)"
    )
    fat_pct: int = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of daily calories allocated to fat (e.g. 25)"
    )

    # Optional persisted targets (preferred when present).
    # When supplied by the backend/DB, the agent should NOT recompute them.
    target_calories: Optional[int] = Field(
        default=None,
        ge=0,
        description="Persisted daily caloric target (kcal). Preferred over recomputation when available.",
    )
    target_protein_g: Optional[int] = Field(
        default=None,
        ge=0,
        description="Persisted daily protein target (g). Preferred over recomputation when available.",
    )
    target_carb_g: Optional[int] = Field(
        default=None,
        ge=0,
        description="Persisted daily carbohydrate target (g). Preferred over recomputation when available.",
    )
    target_fat_g: Optional[int] = Field(
        default=None,
        ge=0,
        description="Persisted daily fat target (g). Preferred over recomputation when available.",
    )

    @model_validator(mode="after")
    def macro_percentages_must_sum_to_100(self) -> UserProfile:
        """Ensures that protein_pct + carb_pct + fat_pct equals exactly 100."""
        total = self.protein_pct + self.carb_pct + self.fat_pct
        if total != 100:
            raise ValueError(
                f"protein_pct + carb_pct + fat_pct must equal 100, got {total}."
            )
        return self

    dietary_style: Optional[DietaryStyle] = Field(
        default=None,
        description=(
            "Optional dietary preference. Exclusion types (VEGAN, VEGETARIAN, CELIAC) cause "
            "non-compliant scanned ingredients to be marked as excluded. Cultural types "
            "(MEDITERRANEAN, ASIAN, ARABIC, LATIN) steer the Chef's recipe-search queries only."
        )
    )
    allergies_and_intolerances: List[str] = Field(
        default_factory=list,
        description=(
            "Known allergies or intolerances as canonical EU-14 English ids, e.g. "
            "['gluten', 'milk', 'peanuts', 'crustaceans', 'molluscs']"
        ),
    )
    barcodes: List[str] = Field(
        ...,
        min_length=1,
        description="EAN/UPC barcodes of products detected by the vision module"
    )

    # Session overrides for adaptive dynamic budgeting
    session_meal_caloric_target: Optional[int] = Field(default=None, ge=0)
    session_meal_protein_g: Optional[int] = Field(default=None, ge=0)
    session_meal_carb_g: Optional[int] = Field(default=None, ge=0)
    session_meal_fat_g: Optional[int] = Field(default=None, ge=0)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class IngredientStatus(str, Enum):
    APPROVED            = "approved"
    SUBSTITUTE_REQUIRED = "substitute_required"
    EXCLUDED            = "excluded"


class CulinaryRole(str, Enum):
    MAIN_PROTEIN   = "main_protein"    # chicken, tuna, eggs, beef, legumes, tofu…
    MAIN_CARB      = "main_carb"       # rice, pasta, bread, potato, oats…
    PRODUCE        = "produce"         # fresh/canned vegetables and fruits
    BASE_CONDIMENT = "base_condiment"  # fats, spices, sauces, seasonings, vinegar…
    SNACK_DRINK    = "snack_drink"     # packaged snacks, confectionery, beverages


class IngredientAssessment(BaseModel):
    barcode: str = Field(
        description="EAN/UPC barcode as scanned by the vision module"
    )
    product_name: str = Field(
        description="Product name as returned by OpenFoodFacts"
    )
    status: IngredientStatus = Field(
        description="Safety assessment result for this ingredient"
    )
    reason: str = Field(
        description="Brief explanation of the status decision"
    )
    substitution_suggestion: Optional[str] = Field(
        default=None,
        description="Concrete alternative ingredient; only populated when status is 'substitute_required'"
    )
    kcal_per_100g: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Caloric density in kcal per 100g, extracted from the OpenFoodFacts profile. "
            "Set to 0.0 if the data is unavailable or the profile returned an error."
        )
    )
    protein_g_per_100g: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Protein in grams per 100g, extracted from the nutritional profile when available. "
            "Estimated only when missing."
        ),
    )
    carbs_g_per_100g: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Carbohydrates in grams per 100g, extracted from the nutritional profile when available. "
            "Estimated only when missing."
        ),
    )
    fat_g_per_100g: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Fat in grams per 100g, extracted from the nutritional profile when available. "
            "Estimated only when missing."
        ),
    )
    culinary_role: CulinaryRole = Field(
        default=CulinaryRole.BASE_CONDIMENT,
        description=(
            "Primary culinary function of this ingredient in a meal context. "
            "Used by the Chef Agent to build semantically meaningful recipe queries."
        )
    )
    generic_name_en: str = Field(
        description=(
            "MANDATORY. Clean English culinary name, stripped of brand names and commercial "
            "adjectives (e.g., 'fresh cheese', 'mango', 'oats', 'olive oil'). "
            "Used DIRECTLY as a RAG recipe search keyword by the Chef Agent. "
            "NEVER leave this as 'unknown ingredient' unless truly unrecognisable."
        )
    )
    generic_name_es: str = Field(
        description=(
            "MANDATORY. Clean Spanish culinary name (e.g., 'queso fresco', 'mango', 'avena', "
            "'aceite de oliva'). Provided as a display label. "
            "NEVER leave this as 'ingrediente desconocido' unless truly unrecognisable."
        )
    )


class NutritionistReport(BaseModel):
    daily_caloric_target: int = Field(
        description="Daily caloric target in kcal (BMR × NEAT-style factors + goal logic), derived from the user profile"
    )
    meal_caloric_target: int = Field(
        description="Per-meal caloric budget in kcal (daily_caloric_target ÷ meals_per_day)"
    )

    meal_protein_g: int = Field(default=0, description="Per-meal protein target in grams")
    meal_carb_g: int = Field(default=0, description="Per-meal carbohydrate target in grams")
    meal_fat_g: int = Field(default=0, description="Per-meal fat target in grams")
    
    daily_protein_g: int = Field(
        description="Daily protein target in grams (daily_caloric_target × protein_pct% ÷ 4 kcal/g)"
    )
    
    daily_carb_g: int = Field(
        description="Daily carbohydrate target in grams (daily_caloric_target × carb_pct% ÷ 4 kcal/g)"
    )
    daily_fat_g: int = Field(
        description="Daily fat target in grams (daily_caloric_target × fat_pct% ÷ 9 kcal/g)"
    )
    ingredient_assessment: List[IngredientAssessment] = Field(
        description="One assessment entry per barcode provided by the vision module"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — CRITIQUE OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class MacroVerdict(str, Enum):
    BALANCED          = "balanced"
    PROTEIN_DEFICIT   = "protein_deficit"
    CARB_DEFICIT      = "carb_deficit"
    FAT_DEFICIT       = "fat_deficit"
    CALORIE_DEFICIT   = "calorie_deficit"
    CALORIE_EXCESS    = "calorie_excess"
    MULTIPLE_DEFICITS = "multiple_deficits"


class CorrectionSuggestion(BaseModel):
    ingredient_name: str = Field(
        description="Generic name of the suggested ingredient (e.g., 'chicken breast', 'lentils')"
    )
    reason: str = Field(
        description=(
            "Brief explanation of which deficit this ingredient addresses "
            "and why it fits the recipe culinarily"
        )
    )


class RecipeCritique(BaseModel):
    recipe_title: str = Field(
        description="Title of the recipe being evaluated, matching the Chef's proposal exactly"
    )
    macro_verdict: MacroVerdict = Field(
        description="Single macro balance verdict for this recipe"
    )
    detected_deficiencies: str = Field(
        description=(
            "Concise explanation of the macro gap(s) with approximate numbers. "
            "Use 'No significant macro deficiencies detected.' when verdict is 'balanced'."
        )
    )
    correction_suggestions: List[CorrectionSuggestion] = Field(
        default_factory=list,
        description=(
            "1–2 generic fresh ingredients to correct severe macro deficits. "
            "Must be an empty list when the recipe is balanced or has calorie_excess."
        )
    )

    @field_validator("correction_suggestions")
    @classmethod
    def cap_suggestions_at_two(
        cls, v: List[CorrectionSuggestion]
    ) -> List[CorrectionSuggestion]:
        """Hard limit: at most 2 correction suggestions per recipe."""
        return v[:2] if len(v) > 2 else v


class NutritionistCritique(BaseModel):
    recipe_critiques: List[RecipeCritique] = Field(
        description="One critique entry per Chef recipe proposal, in the same order as the input"
    )
    overall_recommendation: str = Field(
        description=(
            "Name of the single best-balanced recipe and a 1–2 sentence justification "
            "comparing it against the other two proposals"
        )
    )

    @field_validator("recipe_critiques")
    @classmethod
    def ensure_three_critiques(
        cls, v: List[RecipeCritique]
    ) -> List[RecipeCritique]:
        """
        Ensures exactly 3 critique entries — one per Chef recipe proposal.
        Fewer than 3 is a critical error (missing analysis).
        More than 3 is an LLM hallucination — silently truncate to the first 3.
        """
        if len(v) < 3:
            raise ValueError(
                f"NutritionistCritique must contain at least 3 recipe_critiques, got {len(v)}."
            )
        return v[:3]


# ─────────────────────────────────────────────────────────────────────────────
# CHEAT-MEAL TRACKING SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class CheatMealReport(BaseModel):
    meal_description: str = Field(
        description="The original free-text description of the cheat meal as entered by the user"
    )
    estimated_kcal: int = Field(
        description="Estimated total kilocalories for the full meal (single person)"
    )
    estimated_protein_g: float = Field(
        description="Estimated protein in grams"
    )
    estimated_carb_g: float = Field(
        description="Estimated carbohydrates in grams"
    )
    estimated_fat_g: float = Field(
        description="Estimated fat in grams"
    )
    estimation_notes: str = Field(
        description="Brief note on which items were estimated and confidence level"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — EXACT NUTRITIONAL CALCULATION SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class MacroBreakdown(BaseModel):
    """Per-serving macro totals for a fully corrected final recipe."""
    total_kcal: int = Field(
        description="Total kilocalories per individual serving"
    )
    protein_g: float = Field(
        description="Total protein in grams per serving"
    )
    carb_g: float = Field(
        description="Total carbohydrates in grams per serving"
    )
    fat_g: float = Field(
        description="Total fat in grams per serving"
    )
    accuracy_note: str = Field(
        description=(
            "One sentence stating how many ingredients had exact kcal data and "
            "what was estimated from internal knowledge"
        )
    )
    accuracy_exact_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Ingredients matched to scanned product nutrition data (for i18n).",
    )
    accuracy_estimated_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Ingredients estimated without exact scan data (for i18n).",
    )


class RecipeNutrition(BaseModel):
    recipe_title: str = Field(
        description="Title matching the FinalRecipe exactly"
    )
    macro_breakdown: MacroBreakdown = Field(
        description="Per-serving macro totals for this recipe"
    )


class Phase3Output(BaseModel):
    """Output of the Phase 3 exact nutritional calculation."""
    recipes: List[RecipeNutrition] = Field(
        description="Exactly 3 entries, one per final recipe, in the same order"
    )

    @field_validator("recipes")
    @classmethod
    def must_be_exactly_three(cls, v: List[RecipeNutrition]) -> List[RecipeNutrition]:
        """Enforces exactly 3 nutrition entries — one per final recipe."""
        if len(v) != 3:
            raise ValueError(
                f"Phase3Output must contain exactly 3 recipes, got {len(v)}."
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# THE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class NutritionistAgent:
    """
    Nutritionist Agent: health guard in the multi-agent culinary pipeline.

    Phase 1 — Baseline (run):
      1. Calculates all nutrition targets deterministically via Mifflin-St Jeor (Python/math):
         daily + per-meal caloric budgets and macro gram targets (protein, carbs, fat).
      2. Calls OpenFoodFactsTool once per barcode to fetch nutritional profiles.
      3. Injects all pre-calculated targets and the profiles into the LLM prompt.
      4. Invokes the LLM with structured output for allergen/ingredient assessment.
      5. Returns a NutritionistReport to serve as Base Constraints for the Chef Agent.

    Phase 2 — Critique (evaluate_proposals):
      1. Receives the NutritionistReport and the ChefProposal (3 recipes).
      2. Computes per-serving macros deterministically using stored per-100g profiles when available.
      3. Uses the LLM ONLY to propose up to 2 culinarily coherent correction suggestions when deficits exist.

    All arithmetic is kept in Python so the LLM focuses exclusively on semantic reasoning.
    """

    def __init__(
        self,
        baseline_model: str = config.NUTRITIONIST_LLM_MODEL,
        critique_model: str = config.NUTRITIONIST_CRITIQUE_LLM_MODEL,
    ) -> None:
        print("--- Initializing Nutritionist Agent ---")

        self._tool = OpenFoodFactsTool()

        # Phase 1 uses a faster model — the task is structured and rule-based.
        llm_baseline = ChatOpenAI(model=baseline_model, temperature=0)
        # Phase 1 chain — allergen / dietary assessment → NutritionistReport
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "{user_message}"),
        ])
        self._chain = self._prompt | llm_baseline.with_structured_output(NutritionistReport)

        # Cheat-meal chain — free-text meal → CheatMealReport (fast model is sufficient)
        self._cheat_meal_prompt = ChatPromptTemplate.from_messages([
            ("system", _CHEAT_MEAL_SYSTEM_PROMPT),
            ("human", "Meal description: {description}"),
        ])
        self._cheat_meal_chain = (
            self._cheat_meal_prompt | llm_baseline.with_structured_output(CheatMealReport)
        )


        # Per-ingredient macro estimation (per 100g) for missing ingredients (cached).
        class _MacroEstimate(BaseModel):
            kcal_per_100g: float = Field(ge=0.0)
            protein_g_per_100g: float = Field(ge=0.0)
            carbs_g_per_100g: float = Field(ge=0.0)
            fat_g_per_100g: float = Field(ge=0.0)

        self._macro_estimate_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You estimate nutrition per 100g for a single ingredient. "
                "Return realistic typical values. Avoid extreme outliers. "
                "Ensure consistency: kcal ≈ (protein*4 + carbs*4 + fat*9).",
            ),
            ("human", "Ingredient: {ingredient_name}"),
        ])
        self._macro_estimate_chain = self._macro_estimate_prompt | llm_baseline.with_structured_output(_MacroEstimate)
        self._macro_estimate_cache: dict[str, tuple[float, float, float, float]] = {}

        print("--- Nutritionist Agent Ready ---")

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _calculate_nutrition_targets(
        self, profile: UserProfile
    ) -> Tuple[int, int, int, int, int]:
        """
        Computes all deterministic nutrition targets using the Mifflin-St Jeor formula.
        Keeping this arithmetic in Python ensures reproducible results independent of
        LLM non-determinism or token-budget constraints.

        BMR (male)   = (10 × weight_kg) + (6.25 × height_cm) − (5 × age) + 5
        BMR (female) = (10 × weight_kg) + (6.25 × height_cm) − (5 × age) − 161
        NEAT baseline = BMR × NEAT_multiplier (lifestyle only; not full TDEE)
        Daily target  = NEAT baseline + goal_adjustment
        Meal target  = daily_target ÷ meals_per_day
        Protein (g)  = daily_target × protein_pct% ÷ 4 kcal/g
        Carbs   (g)  = daily_target × carb_pct%    ÷ 4 kcal/g
        Fat     (g)  = daily_target × fat_pct%     ÷ 9 kcal/g

        Returns:
            (daily_caloric_target, meal_caloric_target,
             daily_protein_g, daily_carb_g, daily_fat_g)
        """
        # NEAT-only multipliers (aligned with backend calculate_dynamic_targets).
        _neat_multipliers = {
            ActivityLevel.SEDENTARY:         1.2,
            ActivityLevel.LIGHTLY_ACTIVE:    1.3,
            ActivityLevel.MODERATELY_ACTIVE: 1.4,
            ActivityLevel.VERY_ACTIVE:       1.5,
            ActivityLevel.EXTREMELY_ACTIVE:  1.5,
        }
        _goal_adjustments = {
            PhysicalGoal.WEIGHT_LOSS: -400,
            PhysicalGoal.MAINTENANCE:  0,
            PhysicalGoal.MUSCLE_GAIN: +250,
        }

        if profile.gender == Gender.MALE:
            bmr = (10 * profile.weight_kg) + (6.25 * profile.height_cm) - (5 * profile.age) + 5
        else:
            bmr = (10 * profile.weight_kg) + (6.25 * profile.height_cm) - (5 * profile.age) - 161

        neat_baseline = bmr * _neat_multipliers[profile.activity_level]
        adjusted = neat_baseline + _goal_adjustments[profile.physical_goal]

        daily_target = round_half_up(adjusted)
        meal_target = round_half_up(adjusted / profile.meals_per_day)

        # Macro gram targets derived from the rounded daily caloric target.
        # 1 g protein = 4 kcal | 1 g carbohydrate = 4 kcal | 1 g fat = 9 kcal
        protein_g = round_half_up((daily_target * profile.protein_pct / 100) / 4)
        carb_g = round_half_up((daily_target * profile.carb_pct / 100) / 4)
        fat_g = round_half_up((daily_target * profile.fat_pct / 100) / 9)

        return daily_target, meal_target, protein_g, carb_g, fat_g

    def _gather_nutritional_data(self, profile: UserProfile) -> str:
        """
        Uses a local precomputed JSON mapping for fast macro lookup (no network).

        Falls back to OpenFoodFacts only when a barcode is missing from the local dataset.
        """
        mapping = _get_local_macros_mapping()
        sections: List[str] = []
        for barcode in profile.barcodes:
            rec = mapping.get(barcode)
            if isinstance(rec, dict):
                sections.append(_fmt_local_profile(barcode, rec))
                continue

            # Fallback: only when missing locally (keeps latency bounded)
            print(f" Local miss, querying OpenFoodFacts for barcode: {barcode}")
            profile_text: str = self._tool.invoke({"barcode": barcode})
            sections.append(profile_text)
        return "\n\n".join(sections)

    def _build_user_message(
        self,
        profile: UserProfile,
        nutritional_data: str,
        daily_caloric_target: int,
        meal_caloric_target: int,
        daily_protein_g: int,
        daily_carb_g: int,
        daily_fat_g: int,
    ) -> str:
        """
        Formats the complete human-turn message combining the user's health profile,
        all pre-calculated nutrition targets (injected as immutable facts), and the
        aggregated nutritional data fetched from OpenFoodFacts.
        """
        restrictions = (
            ", ".join(profile.allergies_and_intolerances)
            if profile.allergies_and_intolerances
            else "None declared"
        )
        diet_line = profile.dietary_style.value if profile.dietary_style else "None declared"
        return (
            f"## USER HEALTH PROFILE\n"
            f"Age            : {profile.age} years\n"
            f"Gender         : {profile.gender.value}\n"
            f"Weight         : {profile.weight_kg} kg\n"
            f"Height         : {profile.height_cm} cm\n"
            f"Activity Level : {profile.activity_level.value}\n"
            f"Physical Goal  : {profile.physical_goal.value}\n"
            f"Meals per Day  : {profile.meals_per_day}\n"
            f"Macro Split    : {profile.protein_pct}% protein / "
            f"{profile.carb_pct}% carbs / {profile.fat_pct}% fat\n"
            f"Dietary Style  : {diet_line}\n"
            f"Restrictions   : {restrictions}\n"
            f"\n## PRE-CALCULATED NUTRITION TARGETS\n"
            f"[IMMUTABLE — copy these values verbatim into the corresponding output fields]\n"
            f"Daily Caloric Target : {daily_caloric_target} kcal\n"
            f"Meal Caloric Target  : {meal_caloric_target} kcal  (per meal)\n"
            f"Daily Protein Target : {daily_protein_g} g\n"
            f"Daily Carb Target    : {daily_carb_g} g\n"
            f"Daily Fat Target     : {daily_fat_g} g\n"
            f"\n## NUTRITIONAL PROFILES (from OpenFoodFacts)\n\n"
            f"{nutritional_data}"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, profile: UserProfile) -> NutritionistReport:
        """
        Executes the full nutritionist pipeline and returns a structured report.

        Args:
            profile: Validated UserProfile containing health data and detected barcodes.

        Returns:
            NutritionistReport with caloric target, ingredient assessments,
            macro analysis, and optional shopping suggestions.
        """
        print(f"[Nutritionist] Processing {len(profile.barcodes)} barcode(s)...")

        # Step 1: deterministic arithmetic — no LLM involved.
        # Check if session adaptive targets are provided by the backend API first
        if isinstance(profile.session_meal_caloric_target, int) and profile.session_meal_caloric_target > 0:
            daily_caloric_target = profile.target_calories or 0
            daily_protein_g = profile.target_protein_g or 0
            daily_carb_g = profile.target_carb_g or 0
            daily_fat_g = profile.target_fat_g or 0
            meal_caloric_target = profile.session_meal_caloric_target
            meal_protein_g = profile.session_meal_protein_g or 0
            meal_carb_g = profile.session_meal_carb_g or 0
            meal_fat_g = profile.session_meal_fat_g or 0
        elif (
            isinstance(profile.target_calories, int)
            and isinstance(profile.target_protein_g, int)
            and isinstance(profile.target_carb_g, int)
            and isinstance(profile.target_fat_g, int)
            and profile.target_calories > 0
        ):
            daily_caloric_target = profile.target_calories
            daily_protein_g = profile.target_protein_g
            daily_carb_g = profile.target_carb_g
            daily_fat_g = profile.target_fat_g
            meal_caloric_target = math.floor((daily_caloric_target / profile.meals_per_day) + 0.5)
            meal_protein_g = math.floor((daily_protein_g / profile.meals_per_day) + 0.5)
            meal_carb_g = math.floor((daily_carb_g / profile.meals_per_day) + 0.5)
            meal_fat_g = math.floor((daily_fat_g / profile.meals_per_day) + 0.5)
        else:
            daily_caloric_target, meal_caloric_target, daily_protein_g, daily_carb_g, daily_fat_g = (
                self._calculate_nutrition_targets(profile)
            )
            meal_protein_g = math.floor((daily_protein_g / profile.meals_per_day) + 0.5)
            meal_carb_g = math.floor((daily_carb_g / profile.meals_per_day) + 0.5)
            meal_fat_g = math.floor((daily_fat_g / profile.meals_per_day) + 0.5)

        print(
            f"[Nutritionist] Targets — "
            f"daily: {daily_caloric_target} kcal | "
            f"per meal ({profile.meals_per_day}×): {meal_caloric_target} kcal | "
            f"P: {meal_protein_g}g / C: {meal_carb_g}g / F: {meal_fat_g}g"
        )

        # Step 2: fetch nutritional profiles (local JSON fast path; OFF fallback on miss)
        nutritional_data = self._gather_nutritional_data(profile)

        # Step 3: inject everything into the prompt and call the LLM
        user_message = self._build_user_message(
            profile, nutritional_data,
            daily_caloric_target, meal_caloric_target,
            daily_protein_g, daily_carb_g, daily_fat_g,
        )

        print("[Nutritionist] Invoking LLM for structured analysis...")
        report: NutritionistReport = self._chain.invoke({"user_message": user_message})

        # Deterministically hydrate per-100g macros from the local dataset when available.
        # This makes Phase 2/3 calculations stable and prevents the LLM from being the source of truth
        # for numeric macro values that already exist in `data/product_test_database_macros.json`.
        local_map = _get_local_macros_mapping()

        def _as_float(value: object) -> Optional[float]:
            if isinstance(value, (int, float)):
                return float(value)
            return None

        updated_assessments: List[IngredientAssessment] = []
        for a in report.ingredient_assessment:
            rec = local_map.get(a.barcode)
            if isinstance(rec, dict):
                kcal = _as_float(rec.get("kcal_per_100g"))
                p = _as_float(rec.get("protein_g_per_100g"))
                c = _as_float(rec.get("carbs_g_per_100g"))
                f = _as_float(rec.get("fat_g_per_100g"))
                update: dict = {}
                if kcal is not None:
                    update["kcal_per_100g"] = kcal
                if p is not None:
                    update["protein_g_per_100g"] = p
                if c is not None:
                    update["carbs_g_per_100g"] = c
                if f is not None:
                    update["fat_g_per_100g"] = f
                if update:
                    updated_assessments.append(a.model_copy(update=update))
                    continue
            updated_assessments.append(a)

        # Hydrate all target integers to prevent the LLM from altering computed budgets
        report = report.model_copy(
            update={
                "ingredient_assessment": updated_assessments,
                "daily_caloric_target": daily_caloric_target,
                "meal_caloric_target": meal_caloric_target,
                "meal_protein_g": meal_protein_g,
                "meal_carb_g": meal_carb_g,
                "meal_fat_g": meal_fat_g,
                "daily_protein_g": daily_protein_g,
                "daily_carb_g": daily_carb_g,
                "daily_fat_g": daily_fat_g,
            }
        )
        print("[Nutritionist] Report generated successfully.")
        return report

    def evaluate_proposals(
        self,
        report: NutritionistReport,
        proposal: ChefProposal,
        diners: int = 2,
        output_language: Literal["es", "en"] = "es",
    ) -> NutritionistCritique:
        """
        Phase 2: audits the Chef Agent's recipe proposals for nutritional balance.

        For each of the 3 recipes in the proposal, the LLM:
          - Estimates macro contributions using internal nutritional knowledge.
          - Assigns a MacroVerdict comparing them against per-meal targets derived
            from the NutritionistReport.
          - Suggests 0–2 corrective ingredients if a severe deficit is detected.

        Uses a small LLM call ONLY for correction suggestions when needed,
        independent of the Phase 1 workflow.

        Args:
            report:   The NutritionistReport produced by Phase 1 (provides macro targets).
            proposal: The ChefProposal containing exactly 3 recipes to evaluate.

        Returns:
            NutritionistCritique with one RecipeCritique per recipe and an overall
            recommendation for the downstream Mediator Agent.
        """
        print("[Nutritionist] Phase 2 — auditing Chef proposals (deterministic macros)...")

        # Read exact per-meal macro targets directly from the hydrated report schema
        meal_protein_g = report.meal_protein_g
        meal_carb_g = report.meal_carb_g
        meal_fat_g = report.meal_fat_g

        def _per_serving_totals(recipe: RecipeProposal) -> tuple[int, float, float, float]:
            total_kcal = 0.0
            total_p = 0.0
            total_c = 0.0
            total_f = 0.0
            for ing in list(recipe.available_ingredients) + list(recipe.missing_ingredients):
                profile_100g = self._find_macro_profile(ing.name, report)
                if profile_100g is None:
                    profile_100g = self._estimate_macro_profile(ing.name)
                kcal_100g, p_100g, c_100g, f_100g = profile_100g
                g = float(ing.quantity_g)
                total_kcal += (g * kcal_100g) / 100.0
                total_p += (g * p_100g) / 100.0
                total_c += (g * c_100g) / 100.0
                total_f += (g * f_100g) / 100.0

            per = max(1, int(diners))
            kcal = int(math.floor((total_kcal / per) + 0.5))
            p = round(total_p / per, 1)
            c = round(total_c / per, 1)
            f = round(total_f / per, 1)
            return kcal, p, c, f

        def _verdict(kcal: int, p: float, c: float, f: float) -> tuple[MacroVerdict, str]:
            deficits = []
            if meal_protein_g > 0 and p < 0.5 * meal_protein_g:
                deficits.append("protein")
            if meal_carb_g > 0 and c < 0.5 * meal_carb_g:
                deficits.append("carb")
            if meal_fat_g > 0 and f < 0.5 * meal_fat_g:
                deficits.append("fat")

            if kcal < int(0.6 * report.meal_caloric_target):
                v = MacroVerdict.CALORIE_DEFICIT
            elif kcal > int(1.3 * report.meal_caloric_target):
                v = MacroVerdict.CALORIE_EXCESS
            elif len(deficits) == 0:
                v = MacroVerdict.BALANCED
            elif len(deficits) >= 2:
                v = MacroVerdict.MULTIPLE_DEFICITS
            elif deficits[0] == "protein":
                v = MacroVerdict.PROTEIN_DEFICIT
            elif deficits[0] == "carb":
                v = MacroVerdict.CARB_DEFICIT
            else:
                v = MacroVerdict.FAT_DEFICIT

            if v == MacroVerdict.BALANCED:
                msg = (
                    "No se detectan carencias macro significativas."
                    if output_language == "es"
                    else "No significant macro deficiencies detected."
                )
            else:
                if output_language == "es":
                    msg = (
                        f"Estimación por ración — kcal: {kcal} (objetivo {report.meal_caloric_target}); "
                        f"P: {p}g (objetivo {meal_protein_g}); C: {c}g (objetivo {meal_carb_g}); "
                        f"F: {f}g (objetivo {meal_fat_g})."
                    )
                else:
                    msg = (
                        f"Per-serving estimate — kcal: {kcal} (target {report.meal_caloric_target}); "
                        f"P: {p}g (target {meal_protein_g}); C: {c}g (target {meal_carb_g}); "
                        f"F: {f}g (target {meal_fat_g})."
                    )
            return v, msg

        class _CorrectionsOnly(BaseModel):
            correction_suggestions: List[CorrectionSuggestion] = Field(default_factory=list)

            @field_validator("correction_suggestions")
            @classmethod
            def cap_two(cls, v: List[CorrectionSuggestion]) -> List[CorrectionSuggestion]:
                return v[:2] if len(v) > 2 else v

        corrections_system = (
            "You suggest 0-2 correction ingredients for a recipe given detected macro deficits. "
            "You MUST respect culinary coherence: do not suggest ingredients that clash with the dish. "
            "If no coherent correction exists, return an empty list. "
            "Write the fields `ingredient_name` and `reason` in Spanish."
            if output_language == "es"
            else (
                "You suggest 0-2 correction ingredients for a recipe given detected macro deficits. "
                "You MUST respect culinary coherence: do not suggest ingredients that clash with the dish. "
                "If no coherent correction exists, return an empty list. "
                "Write the fields `ingredient_name` and `reason` in English."
            )
        )
        corrections_prompt = ChatPromptTemplate.from_messages([
            ("system", corrections_system),
            ("human", "{user_message}"),
        ])
        corrections_chain = corrections_prompt | ChatOpenAI(model=config.NUTRITIONIST_CRITIQUE_LLM_MODEL, temperature=0).with_structured_output(_CorrectionsOnly)

        recipe_critiques: List[RecipeCritique] = []
        for recipe in proposal.recipes:
            kcal, p, c, f = _per_serving_totals(recipe)
            v, msg = _verdict(kcal, p, c, f)
            suggestions: List[CorrectionSuggestion] = []
            if v not in (MacroVerdict.BALANCED, MacroVerdict.CALORIE_EXCESS):
                user_message = (
                    f"Recipe title: {recipe.title}\n"
                    f"Per-serving macros: kcal={kcal}, protein_g={p}, carb_g={c}, fat_g={f}\n"
                    f"Targets (per meal): kcal={report.meal_caloric_target}, protein_g={meal_protein_g}, carb_g={meal_carb_g}, fat_g={meal_fat_g}\n"
                    f"Detected deficiencies: {msg}\n"
                    f"Recipe ingredients (available): {[f'{i.name} ({i.quantity_g}g)' for i in recipe.available_ingredients]}\n"
                    f"Recipe ingredients (missing): {[f'{i.name} ({i.quantity_g}g)' for i in recipe.missing_ingredients]}\n"
                )
                out = corrections_chain.invoke({"user_message": user_message})
                suggestions = out.correction_suggestions

            recipe_critiques.append(
                RecipeCritique(
                    recipe_title=recipe.title,
                    macro_verdict=v,
                    detected_deficiencies=msg,
                    correction_suggestions=suggestions,
                )
            )

        # Deterministic overall recommendation: prefer BALANCED, else highest protein ratio vs target.
        best_idx = 0
        best_score = -1e9
        for i, recipe in enumerate(proposal.recipes):
            kcal, p, c, f = _per_serving_totals(recipe)
            v, _ = _verdict(kcal, p, c, f)
            if v == MacroVerdict.BALANCED:
                score = 1e6 + p  # strong preference
            else:
                # score by closeness to targets (simple heuristic)
                score = -abs(p - meal_protein_g) - abs(c - meal_carb_g) - abs(f - meal_fat_g)
            if score > best_score:
                best_score = score
                best_idx = i
        best_title = proposal.recipes[best_idx].title
        if output_language == "es":
            overall = (
                f"Mejor ajuste global de macros: {best_title} "
                f"(según estimaciones deterministas por ración frente a objetivos)."
            )
        else:
            overall = (
                f"Best overall macro fit: {best_title} "
                f"(based on deterministic per-serving macro estimates vs targets)."
            )

        critique = NutritionistCritique(
            recipe_critiques=recipe_critiques,
            overall_recommendation=overall,
        )
        print("[Nutritionist] Critique generated successfully.")
        return critique

    def hydrate_chef_calorie_estimates(
        self,
        report: NutritionistReport,
        proposal: ChefProposal,
        diners: int,
    ) -> ChefProposal:
        """
        Fills `estimated_calories_per_serving` deterministically for each Chef recipe proposal.

        This replaces the LLM-heavy calorie estimation step in the Chef prompt while keeping
        the ChefProposal schema unchanged.
        """
        per = max(1, int(diners))
        updated: list[RecipeProposal] = []
        for recipe in proposal.recipes:
            total_kcal = 0.0
            for ing in list(recipe.available_ingredients) + list(recipe.missing_ingredients):
                profile_100g = self._find_macro_profile(ing.name, report)
                if profile_100g is None:
                    profile_100g = self._estimate_macro_profile(ing.name)
                kcal_100g = float(profile_100g[0])
                total_kcal += (float(ing.quantity_g) * kcal_100g) / 100.0
            per_serving_kcal = int(math.floor((total_kcal / per) + 0.5))
            updated.append(recipe.model_copy(update={"estimated_calories_per_serving": per_serving_kcal}))

        return proposal.model_copy(update={"recipes": updated})

    def estimate_cheat_meal(self, description: str) -> CheatMealReport:
        """
        Phase 0 (cheat-meal path): estimates kcal and macros for a free-text
        meal description supplied by the user.

        This is the terminal operation when `is_cheat_meal = True` — no Chef or
        Mediator nodes are invoked.

        Args:
            description: Free-text description of the indulgent meal
                         (e.g., "double cheeseburger, large fries, Coke").

        Returns:
            CheatMealReport with estimated kcal, protein, carb, fat, and a note.
        """
        print(f"[Nutritionist] Cheat-meal tracking: '{description[:60]}...'")
        report: CheatMealReport = self._cheat_meal_chain.invoke(
            {"description": description}
        )
        print("[Nutritionist] Cheat-meal estimate complete.")
        return report

    def estimate_portion_totals(self, product_name: str, grams: float) -> tuple[float, float, float, float]:
        """
        Total kcal, protein_g, carb_g, fat_g for ``grams`` of ``product_name`` using the
        per-100g LLM macro estimate (cached per ingredient name).
        """
        k100, p100, c100, f100 = self._estimate_macro_profile(product_name)
        factor = float(grams) / 100.0
        return (
            k100 * factor,
            p100 * factor,
            c100 * factor,
            f100 * factor,
        )

    def _find_macro_profile(
        self, ingredient_name: str, report: NutritionistReport
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Returns (kcal, protein, carbs, fat) per 100g by matching the ingredient name against
        NutritionistReport. Uses the same 3-pass strategy as _find_kcal_density, but considers
        both generic_name_en and generic_name_es in token overlap.
        """
        query = ingredient_name.lower().strip()
        query_tokens = {t for t in query.split() if len(t) >= 3}

        def _valid(a: IngredientAssessment) -> bool:
            return (
                a.kcal_per_100g > 0
                or a.protein_g_per_100g > 0
                or a.carbs_g_per_100g > 0
                or a.fat_g_per_100g > 0
            )

        for assessment in report.ingredient_assessment:
            generic_en = assessment.generic_name_en.lower().strip()
            if query == generic_en and _valid(assessment):
                return (
                    float(assessment.kcal_per_100g),
                    float(assessment.protein_g_per_100g),
                    float(assessment.carbs_g_per_100g),
                    float(assessment.fat_g_per_100g),
                )

        for assessment in report.ingredient_assessment:
            tokens = set()
            tokens |= {t for t in assessment.generic_name_en.lower().split() if len(t) >= 3}
            tokens |= {t for t in assessment.generic_name_es.lower().split() if len(t) >= 3}
            if query_tokens & tokens and _valid(assessment):
                return (
                    float(assessment.kcal_per_100g),
                    float(assessment.protein_g_per_100g),
                    float(assessment.carbs_g_per_100g),
                    float(assessment.fat_g_per_100g),
                )

        for assessment in report.ingredient_assessment:
            ref_name = (assessment.substitution_suggestion or assessment.product_name).lower()
            ref_tokens = {t for t in ref_name.split() if len(t) >= 3}
            if query_tokens & ref_tokens and _valid(assessment):
                return (
                    float(assessment.kcal_per_100g),
                    float(assessment.protein_g_per_100g),
                    float(assessment.carbs_g_per_100g),
                    float(assessment.fat_g_per_100g),
                )

        return None

    def _estimate_macro_profile(self, ingredient_name: str) -> tuple[float, float, float, float]:
        key = " ".join(ingredient_name.lower().split())
        cached = self._macro_estimate_cache.get(key)
        if cached is not None:
            return cached
        estimate = self._macro_estimate_chain.invoke({"ingredient_name": ingredient_name})
        profile = (
            float(estimate.kcal_per_100g),
            float(estimate.protein_g_per_100g),
            float(estimate.carbs_g_per_100g),
            float(estimate.fat_g_per_100g),
        )
        self._macro_estimate_cache[key] = profile
        return profile

    def _profile_for_final_ingredient(
        self,
        ing: object,
        nutritionist_report: NutritionistReport,
    ) -> tuple[float, float, float, float]:
        """Same resolution rules as Phase 3: (kcal, P, C, F) per 100g."""
        source_val = getattr(getattr(ing, "source", None), "value", "")
        if source_val == "available":
            profile = self._find_macro_profile(getattr(ing, "name", ""), nutritionist_report)
            if profile is None:
                profile = self._estimate_macro_profile(getattr(ing, "name", ""))
        else:
            profile = self._estimate_macro_profile(getattr(ing, "name", ""))
        return profile

    @staticmethod
    def _dominant_macro_class(p: float, c: float, f: float) -> str:
        """Classify ingredient by which macro contributes the most kcal (4/4/9 rule)."""
        kp, kc, kf = p * 4.0, c * 4.0, f * 9.0
        total = kp + kc + kf
        if total < 1.0:
            return "balanced"
        rp, rc, rf = kp / total, kc / total, kf / total
        if rp >= 0.40 and rp >= rc and rp >= rf:
            return "protein"
        if rc >= 0.40 and rc >= rp and rc >= rf:
            return "carb"
        if rf >= 0.40 and rf >= rp and rf >= rc:
            return "fat"
        return "balanced"

    def _mediator_totals_whole_recipe(
        self,
        ingredients: List[object],
        nutritionist_report: NutritionistReport,
    ) -> tuple[float, float, float, float]:
        """Totals for the full recipe (grams are for all diners together)."""
        total_kcal = total_p = total_c = total_f = 0.0
        for ing in ingredients:
            kcal_100g, p_100g, c_100g, f_100g = self._profile_for_final_ingredient(
                ing, nutritionist_report
            )
            g = float(getattr(ing, "quantity_g", 0))
            total_kcal += (g * kcal_100g) / 100.0
            total_p += (g * p_100g) / 100.0
            total_c += (g * c_100g) / 100.0
            total_f += (g * f_100g) / 100.0
        return total_kcal, total_p, total_c, total_f

    def _mediator_totals_per_serving(
        self,
        ingredients: List[object],
        nutritionist_report: NutritionistReport,
        diners: int,
    ) -> tuple[float, float, float, float]:
        per = max(1, int(diners))
        tk, tp, tc, tf = self._mediator_totals_whole_recipe(ingredients, nutritionist_report)
        return (
            tk / per,
            tp / per,
            tc / per,
            tf / per,
        )

    @staticmethod
    def _within_macro_band(actual: float, target: float, lo: float = 0.85, hi: float = 1.15) -> bool:
        if target <= 0:
            return True
        return lo * target <= actual <= hi * target

    def adjust_mediator_recipes_to_meal_nutrition_targets(
        self,
        mediator_output: "MediatorOutput",
        nutritionist_report: NutritionistReport,
        diners: int = 2,
    ) -> "MediatorOutput":
        """
        Post-Mediator, pre-Phase 3: nudge gram quantities so each recipe's **whole-dish**
        kcal and P/C/F move toward per-meal targets **times diner count** (ingredient
        quantities are totals for the table). Per-serving display is still total ÷ diners.

        Uses selective scaling (protein-dominant / carb-dominant / fat-dominant lines)
        instead of a single global factor on all ingredients, followed by a short joint
        pass if the dish still overshoots the combined calorie/macro bands.
        """
        from src.agents.mediator import FinalIngredient, FinalRecipe, MediatorOutput

        print("[Nutritionist] Post-mediator portion adjustment (macro-aware, deterministic)...")
        report = nutritionist_report
        meals_approx = max(
            1, round(report.daily_caloric_target / max(report.meal_caloric_target, 1))
        )
        # Read exact per-meal allocations from the hydrated report schema
        meal_kcal = int(report.meal_caloric_target)
        meal_protein_g = int(report.meal_protein_g)
        meal_carb_g = int(report.meal_carb_g)
        meal_fat_g = int(report.meal_fat_g)

        per = max(1, int(diners))
        target_kcal_total = float(meal_kcal * per)
        target_p_total = float(meal_protein_g * per)
        target_c_total = float(meal_carb_g * per)
        target_f_total = float(meal_fat_g * per)

        max_iters = 14
        joint_iters = 12
        scale_up = 1.055
        scale_down = 0.945
        scale_mild = 1.035
        scale_mild_down = 0.965

        adjusted_recipes: List[FinalRecipe] = []

        def _macro_class_indices(ing_list: List[FinalIngredient]) -> tuple[List[int], List[int], List[int], List[int]]:
            classes: List[str] = []
            for ing in ing_list:
                _kcal100, pr, cb, ft = self._profile_for_final_ingredient(ing, report)
                classes.append(self._dominant_macro_class(pr, cb, ft))

            def _idx(kind: str) -> List[int]:
                return [i for i, cl in enumerate(classes) if cl == kind]

            idx_p, idx_c, idx_f = _idx("protein"), _idx("carb"), _idx("fat")
            idx_all = list(range(len(ing_list)))
            return idx_p, idx_c, idx_f, idx_all

        for recipe in mediator_output.recipes:
            ingredients: List[FinalIngredient] = [FinalIngredient.model_validate(i.model_dump()) for i in recipe.ingredients]
            orig_qty = [i.quantity_g for i in ingredients]

            for _ in range(max_iters):
                tk, tp, tc, tf = self._mediator_totals_whole_recipe(ingredients, report)
                ok_k = self._within_macro_band(tk, target_kcal_total)
                ok_p = self._within_macro_band(tp, target_p_total)
                ok_c = self._within_macro_band(tc, target_c_total)
                ok_f = self._within_macro_band(tf, target_f_total)
                if ok_k and ok_p and ok_c and ok_f:
                    break

                idx_p, idx_c, idx_f, idx_all = _macro_class_indices(ingredients)

                changed = False
                if meal_protein_g > 0 and tp < 0.85 * target_p_total:
                    if idx_p:
                        for i in idx_p:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_up)
                    else:
                        for i in idx_all:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild)
                    changed = True
                elif meal_carb_g > 0 and tc < 0.85 * target_c_total:
                    if idx_c:
                        for i in idx_c:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_up)
                    else:
                        for i in idx_all:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild)
                    changed = True
                elif meal_fat_g > 0 and tf < 0.85 * target_f_total:
                    if idx_f:
                        for i in idx_f:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_up)
                    else:
                        for i in idx_all:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild)
                    changed = True
                elif tk < 0.85 * target_kcal_total:
                    for i in idx_all:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild)
                    changed = True
                elif meal_protein_g > 0 and tp > 1.15 * target_p_total and idx_p:
                    for i in idx_p:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_down)
                    changed = True
                elif meal_carb_g > 0 and tc > 1.15 * target_c_total and idx_c:
                    for i in idx_c:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_down)
                    changed = True
                elif meal_fat_g > 0 and tf > 1.15 * target_f_total and idx_f:
                    for i in idx_f:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_down)
                    changed = True
                elif tk > 1.15 * target_kcal_total and idx_f:
                    for i in idx_f:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_down)
                    changed = True
                elif tk > 1.15 * target_kcal_total:
                    for i in idx_all:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild_down)
                    changed = True

                if not changed:
                    break

            # Joint pass: whole-dish totals vs combined bands (catches residual kcal overshoot).
            for _ in range(joint_iters):
                tk, tp, tc, tf = self._mediator_totals_whole_recipe(ingredients, report)
                ok_k = self._within_macro_band(tk, target_kcal_total)
                ok_p = self._within_macro_band(tp, target_p_total)
                ok_c = self._within_macro_band(tc, target_c_total)
                ok_f = self._within_macro_band(tf, target_f_total)
                if ok_k and ok_p and ok_c and ok_f:
                    break

                idx_p, idx_c, idx_f, idx_all = _macro_class_indices(ingredients)

                changed = False
                if tk > 1.15 * target_kcal_total:
                    if idx_f:
                        for i in idx_f:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_down)
                    else:
                        for i in idx_all:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild_down)
                    changed = True
                elif tk < 0.85 * target_kcal_total:
                    for i in idx_all:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild)
                    changed = True
                elif meal_protein_g > 0 and tp > 1.15 * target_p_total and idx_p:
                    for i in idx_p:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_down)
                    changed = True
                elif meal_carb_g > 0 and tc > 1.15 * target_c_total and idx_c:
                    for i in idx_c:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_down)
                    changed = True
                elif meal_fat_g > 0 and tf > 1.15 * target_f_total and idx_f:
                    for i in idx_f:
                        ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_down)
                    changed = True
                elif meal_protein_g > 0 and tp < 0.85 * target_p_total:
                    if idx_p:
                        for i in idx_p:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_up)
                    else:
                        for i in idx_all:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild)
                    changed = True
                elif meal_carb_g > 0 and tc < 0.85 * target_c_total:
                    if idx_c:
                        for i in idx_c:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_up)
                    else:
                        for i in idx_all:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild)
                    changed = True
                elif meal_fat_g > 0 and tf < 0.85 * target_f_total:
                    if idx_f:
                        for i in idx_f:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_up)
                    else:
                        for i in idx_all:
                            ingredients[i] = self._scale_final_ingredient_qty(ingredients[i], scale_mild)
                    changed = True

                if not changed:
                    break

            note = recipe.caloric_note
            if any(ingredients[i].quantity_g != orig_qty[i] for i in range(len(ingredients))):
                note = (
                    f"{note} Portions were adjusted toward your per-meal calorie and macro targets "
                    f"(whole dish ≈ {meal_kcal} kcal × {per} diners)."
                ).strip()

            adjusted_recipes.append(
                recipe.model_copy(
                    update={
                        "ingredients": ingredients,
                        "caloric_note": note,
                    }
                )
            )

        return MediatorOutput(recipes=adjusted_recipes)

    @staticmethod
    def _scale_final_ingredient_qty(ing: object, factor: float) -> object:
        g = int(getattr(ing, "quantity_g", 0))
        new_g = max(5, min(4000, int(round(g * factor))))
        if new_g == g:
            if factor > 1.0:
                new_g = min(4000, g + max(1, int(round(g * 0.02)) or 1))
            else:
                new_g = max(5, g - max(1, int(round(g * 0.02)) or 1))
        return ing.model_copy(update={"quantity_g": new_g})  # type: ignore[union-attr]

    def calculate_final_macros(
        self,
        mediator_output: "MediatorOutput",  # forward ref — avoids circular import
        nutritionist_report: NutritionistReport,
        diners: int = 2,
    ) -> Phase3Output:
        """
        Phase 3: computes exact per-serving macros for all 3 final recipes.

        Ingredient quantities are **whole-recipe** totals (same convention as the Mediator);
        this method sums macros for the full recipe, then divides by `diners` for displayed
        per-serving values.

        Strategy:
          - Available ingredients: kcal anchored to real OpenFoodFacts density
            (looked up via token matching against NutritionistReport assessments).
          - To-buy and nutritionist-addition ingredients: LLM estimates from
            internal nutritional knowledge.

        Args:
            mediator_output:       The 3 corrected recipes from the Mediator.
            nutritionist_report:   Phase 1 report carrying kcal_per_100g per barcode.
            diners:                Diner count for per-serving division (totals ÷ diners).

        Returns:
            Phase3Output with a MacroBreakdown per recipe.
        """
        print("[Nutritionist] Phase 3 — calculating macros for final recipes (deterministic)...")

        per = max(1, int(diners))
        results: List[RecipeNutrition] = []
        for recipe in mediator_output.recipes:
            total_kcal = 0.0
            total_p = 0.0
            total_c = 0.0
            total_f = 0.0
            exact_count = 0
            estimated_count = 0

            for ing in recipe.ingredients:
                source_val = ing.source.value  # "available" / "to_buy" / "nutritionist_addition"
                profile_100g: Optional[tuple[float, float, float, float]]

                if source_val == "available":
                    profile_100g = self._find_macro_profile(ing.name, nutritionist_report)
                    if profile_100g is not None:
                        exact_count += 1
                    else:
                        profile_100g = self._estimate_macro_profile(ing.name)
                        estimated_count += 1
                else:
                    profile_100g = self._estimate_macro_profile(ing.name)
                    estimated_count += 1

                kcal_100g, p_100g, c_100g, f_100g = profile_100g
                g = float(ing.quantity_g)
                total_kcal += (g * kcal_100g) / 100.0
                total_p += (g * p_100g) / 100.0
                total_c += (g * c_100g) / 100.0
                total_f += (g * f_100g) / 100.0

            per_kcal = int(math.floor((total_kcal / per) + 0.5))
            per_p = round(total_p / per, 1)
            per_c = round(total_c / per, 1)
            per_f = round(total_f / per, 1)

            accuracy = (
                f"Exact macro profile matched for {exact_count} scanned ingredient(s); "
                f"remaining {estimated_count} ingredient(s) estimated."
            )

            results.append(
                RecipeNutrition(
                    recipe_title=recipe.title,
                    macro_breakdown=MacroBreakdown(
                        total_kcal=per_kcal,
                        protein_g=per_p,
                        carb_g=per_c,
                        fat_g=per_f,
                        accuracy_note=accuracy,
                        accuracy_exact_count=exact_count,
                        accuracy_estimated_count=estimated_count,
                    ),
                )
            )

        output = Phase3Output(recipes=results)
        print("[Nutritionist] Phase 3 complete.")
        return output


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # ChefProposal lives in chef.py which imports from this module,
    # so the import is deferred to __main__ to avoid circular imports.
    from src.agents.chef import ChefProposal, RecipeIngredient, RecipeProposal

    agent = NutritionistAgent()

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    # Simulates what the vision module would pass after scanning a user's fridge.
    # Nutella should trigger substitute_required (lactose intolerance).
    # Macro split: 30% protein / 45% carbs / 25% fat (must sum to 100).
    test_profile = UserProfile(
        age=30,
        gender=Gender.MALE,
        weight_kg=80.0,
        height_cm=180.0,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        physical_goal=PhysicalGoal.MAINTENANCE,
        meals_per_day=3,
        protein_pct=30,
        carb_pct=45,
        fat_pct=25,
        allergies_and_intolerances=[],
        barcodes=[
            "3017620422003",   # Nutella (contains milk — should trigger substitute_required)
            "3155250349793",   # Harina de trigo (wheat flour — approved)
            "00000000000000",  # Non-existent — should produce an EXCLUDED entry
        ],
    )

    print("\n" + "=" * 60)
    print("NUTRITIONIST AGENT — PHASE 1: BASELINE")
    print("=" * 60)

    report = agent.run(test_profile)

    print("\n--- STRUCTURED REPORT ---")
    print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    # Synthetic ChefProposal that exercises different deficit scenarios:
    #   Recipe 1 — "Pasta al Tomate": carb-heavy, no protein source → protein_deficit
    #   Recipe 2 — "Tortilla Española": balanced (eggs + potato + olive oil)
    #   Recipe 3 — "Crema de Verduras": low-calorie vegetable soup → calorie_deficit
    mock_proposal = ChefProposal(
        recipes=[
            RecipeProposal(
                title="Pasta al Tomate con Aceite de Oliva",
                justification="High carb overlap; caloric fit is approximate.",
                estimated_calories_per_serving=480,
                available_ingredients=[
                    RecipeIngredient(name="wheat flour pasta", quantity_g=150),
                    RecipeIngredient(name="tomato sauce",      quantity_g=120),
                    RecipeIngredient(name="olive oil",         quantity_g=20),
                ],
                missing_ingredients=[],
                preparation_steps=[
                    "Boil pasta until al dente.",
                    "Warm tomato sauce in a pan.",
                    "Combine and drizzle with olive oil.",
                ],
            ),
            RecipeProposal(
                title="Tortilla Española",
                justification="Uses eggs, potato, and olive oil — strong macro coverage.",
                estimated_calories_per_serving=520,
                available_ingredients=[
                    RecipeIngredient(name="eggs",      quantity_g=200),
                    RecipeIngredient(name="olive oil", quantity_g=30),
                ],
                missing_ingredients=[
                    RecipeIngredient(name="potato", quantity_g=300),
                    RecipeIngredient(name="onion",  quantity_g=80),
                ],
                preparation_steps=[
                    "Slice potatoes and onion; fry gently in olive oil.",
                    "Beat eggs, season with salt, mix with potatoes.",
                    "Cook tortilla on both sides until set.",
                ],
            ),
            RecipeProposal(
                title="Crema de Verduras",
                justification="Light vegetable soup; may require caloric check.",
                estimated_calories_per_serving=180,
                available_ingredients=[
                    RecipeIngredient(name="olive oil", quantity_g=15),
                ],
                missing_ingredients=[
                    RecipeIngredient(name="zucchini", quantity_g=200),
                    RecipeIngredient(name="carrot",   quantity_g=100),
                    RecipeIngredient(name="onion",    quantity_g=80),
                    RecipeIngredient(name="vegetable broth", quantity_g=400),
                ],
                preparation_steps=[
                    "Sauté onion in olive oil.",
                    "Add diced zucchini and carrot; cook 5 minutes.",
                    "Pour in broth, simmer 20 minutes, then blend.",
                ],
            ),
        ]
    )

    print("NUTRITIONIST AGENT — PHASE 2: CRITIQUE")

    critique = agent.evaluate_proposals(report, mock_proposal)

    print("\n NUTRITIONIST CRITIQUE")
    print(json.dumps(critique.model_dump(), indent=2, ensure_ascii=False))
