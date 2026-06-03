from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load the project .env before any LangChain / OpenAI client is instantiated,
# so OPENAI_API_KEY is available from the moment the module is imported.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

from src.agents.tools.chef_recipe_tool import (
    ChefRecipeSearchTool,
    _format_candidates_as_text,
    _parse_stored_list,
)
from src.agents.nutritionist import (
    CulinaryRole,
    DietaryStyle,
    IngredientAssessment,
    IngredientStatus,
    NutritionistReport,
)
from src.agents import config


# ─────────────────────────────────────────────────────────────────────────────
# NAME NORMALIZATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _core_name(product_name: str) -> str:
    """
    Extracts the first 1–2 semantically meaningful words from a commercial
    OpenFoodFacts product name so they can be used as RAG search terms.

    Examples:
      "Queso fresco de burgos oveja"  →  "queso fresco"
      "MANGO kent"                    →  "mango kent"
      "Aceite de oliva virgen extra"  →  "aceite"
      "Harina de trigo integral"      →  "harina"
      "sparkling water lemon"         →  "sparkling water"

    Strategy: lowercase, then collect words until we hit a stop-word (preposition /
    article) — but only stop after we have at least one word — or until we have 2 words.
    """
    words = product_name.lower().split()
    result: list[str] = []
    for word in words:
        if word in _STOP_AT_WORDS and result:
            break
        result.append(word)
        if len(result) >= 2:
            break
    return " ".join(result) if result else product_name.lower()


# ─────────────────────────────────────────────────────────────────────────────
# PANTRY AND RETRIEVAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Household staples assumed to be in every kitchen.
# Used both for the deterministic availability filter and as "already available"
# context injected into the LLM prompt.
_BASE_PANTRY: frozenset[str] = frozenset({
    # English
    "salt", "pepper", "black pepper", "white pepper", "olive oil",
    "extra virgin olive oil", "vegetable oil", "sunflower oil",
    "water", "garlic", "onion", "vinegar", "white vinegar",
    "red wine vinegar", "sugar", "flour", "all-purpose flour",
    "butter", "baking soda", "baking powder",
    # Spanish equivalents
    "sal", "pimienta", "pimienta negra", "aceite de oliva",
    "aceite de oliva virgen extra", "aceite vegetal", "aceite de girasol",
    "agua", "ajo", "cebolla", "vinagre", "vinagre blanco",
    "vinagre de vino tinto", "azúcar", "harina", "harina de trigo",
    "mantequilla", "bicarbonato", "levadura",
})

# Concise English-only display list injected into the LLM prompt
_BASE_PANTRY_DISPLAY: list[str] = [
    "salt", "black pepper", "olive oil", "water",
    "garlic", "onion", "vinegar", "sugar", "flour", "butter",
]

# Raw candidates fetched per query (large pool → better coverage after ranking)
_RAW_FETCH_PER_QUERY: int = 30

# Words that mark the end of the meaningful part of a Spanish/English product name.
# Used by _core_name() to truncate "Queso fresco de burgos oveja" → "queso fresco".
_STOP_AT_WORDS: frozenset[str] = frozenset({
    "de", "del", "con", "y", "e", "a", "al", "el", "la", "los", "las",
    "en", "sin", "para", "por", "of", "with", "and", "from", "in",
})

# Cuisine-characteristic keyword lists.  Exclusion diets (vegan, vegetarian, celiac)
# have no entry — they steer retrieval through Nutritionist exclusions, not query injection.
# Each entry is a SHORT list of individual keywords; the query builder picks ONE keyword
# per query (cycling) to avoid semantic dilution while still steering the embedding space.
_CUISINE_SIGNATURES: dict[str, list[str]] = {
    "mediterranean": ["tomato", "lemon", "herbs", "olive"],
    "asian":         ["ginger", "sesame", "soy", "wok"],
    "arabic":        ["cumin", "chickpeas", "tahini", "coriander"],
    "latin":         ["beans", "lime", "cilantro", "chili"],
}


def _culinary_keyword_en(assessment: IngredientAssessment) -> str:
    """English RAG keyword — prefer generic_name_en over substitution text."""
    return (assessment.generic_name_en or assessment.substitution_suggestion or "").strip()


def _culinary_keyword_es(assessment: IngredientAssessment) -> str:
    """Spanish RAG keyword — prefer generic_name_es."""
    return (
        assessment.generic_name_es
        or assessment.generic_name_en
        or assessment.substitution_suggestion
        or ""
    ).strip()


def _priority_tokens(assessment: IngredientAssessment) -> set[str]:
    """Normalized tokens used to detect the pinned ingredient in text."""
    tokens: set[str] = set()
    for raw in (
        assessment.generic_name_en,
        assessment.generic_name_es,
        assessment.substitution_suggestion,
    ):
        if isinstance(raw, str):
            t = raw.lower().strip()
            if len(t) >= 3:
                tokens.add(t)
    return tokens


def _text_has_priority_token(text: str, tokens: set[str]) -> bool:
    if not tokens:
        return True
    lowered = text.lower()
    return any(
        len(token) >= 3 and (token in lowered or lowered in token)
        for token in tokens
    )


def _recipe_dict_has_priority(recipe: dict, tokens: set[str]) -> bool:
    combined = (
        f"{recipe.get('title', '')} {recipe.get('ingredients', '')}"
    ).lower()
    return _text_has_priority_token(combined, tokens)


def _inject_priority_if_missing(query: str, token: str, all_tokens: set[str]) -> str:
    """Prepend one keyword only when no priority variant is already in the query."""
    q = " ".join(query.split())
    if not token or _text_has_priority_token(q, all_tokens):
        return q
    return f"{token} {q}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Chef Agent — the "Actor" in the multi-agent culinary assistant pipeline.

You receive:
  1. BASE CONSTRAINTS from the Nutritionist Agent (caloric + macro targets,
     approved ingredients with caloric density, excluded allergen/dietary ingredients).
  2. BASE PANTRY — universal household staples (salt, pepper, olive oil, garlic,
     onion, vinegar, flour, butter…) that the user is ASSUMED TO HAVE.
  3. DIETARY PREFERENCE (optional) — the user's dietary style if declared.
  4. NUMBER OF DINERS to scale every recipe for.
  5. CANDIDATE RECIPES retrieved from the recipe knowledge base and pre-ranked
     in Python by ingredient coverage (highest overlap with the user's approved
     list + base pantry first). Candidate 1 has the best coverage; later
     candidates require progressively more items to be purchased.

Your mission: propose EXACTLY 3 complete, adapted, cook-ready recipes.
Complete the following five steps in order.

────────────────────────────────────────────
STEP 1 — Recipe Selection
────────────────────────────────────────────
From the candidate pool, choose the 3 BEST options. Rank by:
  a) Maximum overlap between the recipe's required ingredients and the APPROVED ingredients.
  b) Estimated calories per serving within ±20% of `meal_caloric_target`.
  c) ZERO presence of EXCLUDED ingredients (allergens or dietary violations) — this rule is
     absolute and overrides all other criteria.
  d) DIETARY PREFERENCE compliance:
       • Cultural styles (mediterranean, asian, arabic, latin): prefer recipes that reflect the
         declared cuisine's flavour profile and characteristic ingredients.
       • Exclusion styles (vegan, vegetarian, celiac): treat all excluded ingredients exactly
         as allergens — reject any recipe that requires them, even as missing ingredients.
  e) USER PRIORITY INGREDIENT (when declared in the human message):
       The user pinned ONE ingredient that MUST appear structurally in every recipe's
       `available_ingredients` (scaled grams, not optional garnish). Prefer candidates
       that already contain it; when adapting, integrate it into preparation_steps.

If fewer than 3 candidates are sufficiently suitable, adapt the closest available matches.

────────────────────────────────────────────
STEP 3 — Convert All Quantities to Grams
────────────────────────────────────────────
Every ingredient quantity in your output MUST be expressed in GRAMS (integer).
No cups, oz, tbsp, tsp, or any other unit is permitted in the output.

Use your internal culinary knowledge to convert every unit you encounter to grams,
including non-standard or colloquial quantities in both English and Spanish:
  cups, oz, ounces, lbs, tbsp, tablespoons, tsp, teaspoons,
  "a pinch", "a dash", "to taste", "a handful", "a drizzle",
  "una pizca", "un chorrito", "al gusto", "un puñado", "una cucharada",
  count-based items ("1 egg", "2 cloves garlic", "1 medium onion"), and any other
  unit or description present in the recipe data.

Round every final value to the nearest integer gram.

────────────────────────────────────────────
STEP 4 — Classify Ingredients
────────────────────────────────────────────
For each selected recipe, split the scaled + converted ingredients into two groups:

  available_ingredients:
    An ingredient may appear here ONLY if it satisfies AT LEAST ONE of the following:
      • It matches (exactly or as a close translation) an item in the APPROVED INGREDIENTS list.
      • It matches a BASE PANTRY item (salt, pepper, olive oil, garlic, onion, vinegar,
        flour, butter, sugar, water, baking soda, baking powder).

  missing_ingredients:
    Every other recipe ingredient that is STRUCTURALLY VITAL (the dish cannot be completed
    without it) and is NOT covered by the approved list or base pantry.
    Optional garnishes, "to taste" seasonings, and base pantry staples MUST NOT appear here.

  ⚠ CRITICAL — ANTI-HALLUCINATION RULE (NON-NEGOTIABLE):
    You are STRICTLY FORBIDDEN from placing any ingredient in `available_ingredients`
    if it is not explicitly present in the APPROVED INGREDIENTS list or BASE PANTRY
    provided in the context. Do NOT infer, assume, or invent availability.
    If a recipe calls for lamb, chicken, rice, pasta, or any other ingredient that
    does NOT appear in the approved list or base pantry, it MUST go to
    `missing_ingredients`. Inventing availability corrupts the user's shopping list
    and is a critical system failure.

────────────────────────────────────────────
OUTPUT RULES (NON-NEGOTIABLE)
────────────────────────────────────────────
  • The `recipes` list MUST contain EXACTLY 3 entries — no more, no fewer.
  • ALL ingredient quantities MUST be integers representing grams.
  • All quantities MUST already be scaled for the provided DINERS value.
  • `preparation_steps` must be clear, numbered, actionable cooking steps adapted to
    the available ingredients and the scaled quantities.
  • `justification` must explain: (a) ingredient overlap with approved list,
    (b) caloric fit relative to `meal_caloric_target`, (c) allergen safety confirmation.

The user's human message includes an OUTPUT LANGUAGE block. All user-visible fields
(recipe titles, ingredient names, preparation_steps, justification) MUST follow
that language — do not mix languages.
"""


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class ChefRequest(BaseModel):
    nutritionist_report: NutritionistReport = Field(
        description="The Base Constraints report produced by the Nutritionist Agent"
    )
    diners: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of people to serve; all recipe quantities are scaled to this value"
    )
    dietary_style: Optional[DietaryStyle] = Field(
        default=None,
        description=(
            "Optional dietary preference forwarded from UserProfile. "
            "Cultural types steer RAG queries; exclusion types add an extra reject filter."
        )
    )
    available_minutes: Optional[int] = Field(
        default=None,
        ge=5,
        le=240,
        description=(
            "User's available cooking time in minutes. Converted to a max_ingredients "
            "ceiling via the time-to-complexity heuristic before RAG retrieval."
        )
    )
    dish_type_filter: Optional[str] = Field(
        default=None,
        description=(
            "Optional dish-type metadata filter forwarded from session context "
            "(e.g. 'soup_stew', 'salad_side', 'dessert', 'main_course'). "
            "Applied as an exact-match pre-filter in ChromaDB before semantic ranking."
        )
    )
    output_language: Literal["es", "en"] = Field(
        default="es",
        description="Language for all user-facing recipe text (titles, steps, notes).",
    )
    priority_barcode: Optional[str] = Field(
        default=None,
        description=(
            "Optional EAN/UPC of a user-pinned ingredient. When set, its culinary-role "
            "keyword replaces the default first pick in RAG search queries."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class RecipeIngredient(BaseModel):
    name: str = Field(
        description="Ingredient name"
    )
    quantity_g: int = Field(
        description="Scaled quantity in grams (after unit conversion and diner scaling)"
    )


class RecipeProposal(BaseModel):
    title: str = Field(
        description="Recipe title as retrieved from the knowledge base"
    )
    justification: str = Field(
        description=(
            "Why this recipe was selected: ingredient overlap with approved list, "
            "caloric fit vs meal_caloric_target, and allergen safety confirmation"
        )
    )
    estimated_calories_per_serving: int = Field(
        description="Estimated kcal for one individual serving of the scaled recipe"
    )
    available_ingredients: List[RecipeIngredient] = Field(
        description="Recipe ingredients that the user physically has (from the approved list)"
    )
    missing_ingredients: List[RecipeIngredient] = Field(
        description=(
            "Structurally vital ingredients the recipe requires that are NOT in the approved list. "
            "Garnishes and optional spices must NOT appear here."
        )
    )
    preparation_steps: List[str] = Field(
        description="Numbered, actionable cooking steps adapted to available ingredients and scaled quantities"
    )


class ChefProposal(BaseModel):
    recipes: List[RecipeProposal] = Field(
        description="Exactly 3 adapted recipe proposals"
    )

    @field_validator("recipes")
    @classmethod
    def must_be_exactly_three(cls, v: List[RecipeProposal]) -> List[RecipeProposal]:
        """Hard constraint: the Chef must always propose exactly 3 recipes."""
        if len(v) != 3:
            raise ValueError(
                f"ChefProposal must contain exactly 3 recipes, got {len(v)}."
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# FAST SELECTION SCHEMA (small structured output)
# ─────────────────────────────────────────────────────────────────────────────

class ChefSelection(BaseModel):
    """Small output to avoid long structured generations."""
    selected_indexes: List[int] = Field(
        description="Exactly 3 unique candidate indices (1-based) to use."
    )
    rationale: str = Field(description="Brief reason for the selection.")

    @field_validator("selected_indexes")
    @classmethod
    def must_be_three_unique(cls, v: List[int]) -> List[int]:
        if len(v) != 3:
            raise ValueError("selected_indexes must have length 3.")
        if len(set(v)) != 3:
            raise ValueError("selected_indexes must be unique.")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# THE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ChefAgent:
    """
    Chef Agent: the "Actor" in the multi-agent culinary pipeline.

    Retrieval pipeline (fully deterministic — no LLM):
      1. Extract approved / excluded ingredient lists from the Nutritionist report.
      2. Build 2–3 culinary-role-aware queries using _core_name() to strip
         commercial product names down to their essential culinary term.
      3. Over-fetch: up to _RAW_FETCH_PER_QUERY results per query, deduplicated.
      4. Soft coverage ranking: score every candidate by the fraction of its
         ingredients covered by the user's pantry + base pantry; sort descending.
         No hard cutoff — the best available candidates always reach the LLM.
      5. Pass the top _MAX_CANDIDATES to the LLM.

    LLM reasoning phase:
      6. Select, scale, unit-convert, and estimate calories for exactly 3 recipes.
      7. Anti-hallucination rule enforced in prompt: available_ingredients may only
         contain items explicitly listed in the approved or base pantry lists.
      8. Return a ChefProposal for the downstream Nutritionist Phase 2 / Mediator.
    """

    # Maximum candidates passed to the LLM after coverage ranking
    _MAX_CANDIDATES: int = 6

    def __init__(self, model: str = config.CHEF_LLM_MODEL) -> None:
        print("--- Initializing Chef Agent ---")

        self._tool = ChefRecipeSearchTool()

        llm = ChatOpenAI(model=model, temperature=0)
        self._structured_llm = llm.with_structured_output(ChefProposal)

        # Stage 1: tiny selector chain (fast and avoids huge structured outputs)
        selector_llm = ChatOpenAI(model=model, temperature=0)
        self._selector_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are selecting 3 recipes from a ranked candidate list.\n"
                "Return exactly 3 unique candidate indices (1-based) that best match the pantry.\n"
                "When the message includes USER PRIORITY INGREDIENT, every selected recipe MUST "
                "contain that ingredient in its ingredient list (English or Spanish name). "
                "Never pick 3 recipes that omit the pinned ingredient when alternatives exist.\n"
                "Keep the rationale brief.",
            ),
            ("human", "{user_message}"),
        ])
        self._selector_chain = self._selector_prompt | selector_llm.with_structured_output(ChefSelection)

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "{user_message}"),
        ])

        # The chain: format the prompt → call the structured LLM → parse into ChefProposal
        self._chain = self._prompt | self._structured_llm

        print("--- Chef Agent Ready ---")

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_ingredient_lists(
        self, report: NutritionistReport
    ) -> tuple[list[IngredientAssessment], list[str]]:
        """
        Separates the nutritionist's assessments into two groups:
          - approved_assessments: full IngredientAssessment objects for ingredients
            the user has (APPROVED) or can substitute (SUBSTITUTE_REQUIRED).
            Carrying the full object preserves kcal_per_100g for caloric calculation.
          - excluded_names: plain name strings for allergen ingredients (EXCLUDED).

        Returns:
            (approved_assessments, excluded_names)
        """
        approved: list[IngredientAssessment] = []
        excluded: list[str] = []

        for assessment in report.ingredient_assessment:
            if assessment.status in (
                IngredientStatus.APPROVED,
                IngredientStatus.SUBSTITUTE_REQUIRED,
            ):
                approved.append(assessment)
            elif assessment.status == IngredientStatus.EXCLUDED:
                excluded.append(assessment.product_name)

        return approved, excluded

    @staticmethod
    def _find_priority_assessment(
        approved_assessments: list[IngredientAssessment],
        priority_barcode: Optional[str],
    ) -> Optional[IngredientAssessment]:
        if not priority_barcode:
            return None
        for assessment in approved_assessments:
            if assessment.barcode == priority_barcode:
                return assessment
        print(
            f"[Chef] Priority barcode {priority_barcode!r} not in approved list; "
            "ignoring priority pin."
        )
        return None

    def _build_search_queries(
        self,
        approved_assessments: list[IngredientAssessment],
        dietary_style: Optional[DietaryStyle] = None,
        priority_barcode: Optional[str] = None,
    ) -> list[str]:
        """
        Builds 2–3 role-diverse queries plus an optional 4th priority-only anchor.

        Strategy: fewer queries, each with more varied roles (protein/carb/produce/fat).
        This increases semantic specificity while reducing the total retrieval work.
        """
        queries: list[str] = []
        query_locales: list[Literal["en", "es"]] = []

        def _english_keyword(a: IngredientAssessment) -> str:
            return a.substitution_suggestion or a.generic_name_en

        def _spanish_keyword(a: IngredientAssessment) -> str:
            return a.generic_name_es or _english_keyword(a)

        def _first(items: list[str]) -> Optional[str]:
            return items[0] if items else None

        priority_assessment = self._find_priority_assessment(
            approved_assessments, priority_barcode
        )

        def _pick_role_keyword(
            role: CulinaryRole,
            bucket: list[str],
            keyword_fn,
        ) -> Optional[str]:
            if (
                priority_assessment is not None
                and priority_assessment.culinary_role == role
            ):
                return keyword_fn(priority_assessment)
            return _first(bucket)

        def _collect_keywords(
            keyword_fn,
        ) -> tuple[list[str], list[str], list[str], list[str]]:
            proteins_local: list[str] = []
            carbs_local: list[str] = []
            produce_local: list[str] = []
            fats_local: list[str] = []

            for assessment in approved_assessments:
                kw = keyword_fn(assessment)
                if assessment.culinary_role == CulinaryRole.MAIN_PROTEIN:
                    proteins_local.append(kw)
                elif assessment.culinary_role == CulinaryRole.MAIN_CARB:
                    carbs_local.append(kw)
                elif assessment.culinary_role == CulinaryRole.PRODUCE:
                    produce_local.append(kw)
                elif assessment.culinary_role == CulinaryRole.BASE_CONDIMENT:
                    fats_local.append(kw)

            return proteins_local, carbs_local, produce_local, fats_local

        proteins_en, carbs_en, produce_en, fats_en = _collect_keywords(_english_keyword)
        proteins_es, carbs_es, produce_es, fats_es = _collect_keywords(_spanish_keyword)

        fat_en = _pick_role_keyword(CulinaryRole.BASE_CONDIMENT, fats_en, _english_keyword)
        protein_en = _pick_role_keyword(CulinaryRole.MAIN_PROTEIN, proteins_en, _english_keyword)
        carb_en = _pick_role_keyword(CulinaryRole.MAIN_CARB, carbs_en, _english_keyword)
        produce_item_en = _pick_role_keyword(CulinaryRole.PRODUCE, produce_en, _english_keyword)

        fat_spanish = _pick_role_keyword(CulinaryRole.BASE_CONDIMENT, fats_es, _spanish_keyword)
        protein_spanish = _pick_role_keyword(
            CulinaryRole.MAIN_PROTEIN, proteins_es, _spanish_keyword
        )
        carb_spanish = _pick_role_keyword(CulinaryRole.MAIN_CARB, carbs_es, _spanish_keyword)
        produce_item_spanish = _pick_role_keyword(
            CulinaryRole.PRODUCE, produce_es, _spanish_keyword
        )

        # 1) Balanced anchor: protein + carb + produce + fat
        if protein_en and carb_en:
            parts = [protein_en, carb_en]
            if produce_item_en:
                parts.append(produce_item_en)
            parts.append(fat_en)
            queries.append(" ".join(p for p in parts[:6] if p))
            query_locales.append("en")

        # 2) Protein-forward: protein + produce + fat
        if protein_en and produce_item_en:
            queries.append(" ".join(p for p in [protein_en, produce_item_en, fat_en] if p))
            query_locales.append("en")

        # 3) Spanish anchor: protein + carb + produce + fat (Spanish)
        if protein_spanish and carb_spanish:
            parts_es = [protein_spanish, carb_spanish]
            if produce_item_spanish:
                parts_es.append(produce_item_spanish)
            parts_es.append(fat_spanish)
            queries.append(" ".join(p for p in parts_es[:6] if p))
            query_locales.append("es")

        # Fallback: build one broader query from up to 4 non-snack roles.
        if not queries:
            broad_keywords: list[str] = []
            for assessment in approved_assessments:
                if assessment.culinary_role == CulinaryRole.SNACK_DRINK:
                    continue
                broad_keywords.append(_english_keyword(assessment))
                if len(broad_keywords) >= 4:
                    break
            if broad_keywords:
                queries.append(" ".join(broad_keywords))
                query_locales.append("en")

        # Final fallback: absolutely no meaningful roles classified
        if not queries:
            fallback_keywords: list[str] = []
            for assessment in approved_assessments[:3]:
                fallback_keywords.append(_english_keyword(assessment))
            queries = [" ".join(fallback_keywords) or "quick healthy meal recipe"]
            query_locales = ["en"]

        # Deduplicate role queries (keep locale aligned)
        deduped: list[str] = []
        deduped_locales: list[Literal["en", "es"]] = []
        for q, loc in zip(queries, query_locales):
            qv = " ".join(q.split())
            if qv and qv not in deduped:
                deduped.append(qv)
                deduped_locales.append(loc)
        queries = deduped[:3]
        query_locales = deduped_locales[:3]

        # Language-aware priority injection (skip if any priority token already present)
        if priority_assessment is not None:
            pin_tokens = _priority_tokens(priority_assessment)
            pin_kw_en = _culinary_keyword_en(priority_assessment)
            pin_kw_es = _culinary_keyword_es(priority_assessment)
            print(f"[Chef] Priority tokens for RAG: {sorted(pin_tokens)}")

            for i, q in enumerate(queries):
                locale = query_locales[i] if i < len(query_locales) else "en"
                pin_token = pin_kw_es if locale == "es" else pin_kw_en
                queries[i] = _inject_priority_if_missing(q, pin_token, pin_tokens)

            # 4th query: bilingual anchor for the pinned ingredient only
            dedicated_parts: list[str] = []
            for part in (pin_kw_en, pin_kw_es):
                part_norm = part.strip()
                if part_norm and part_norm.lower() not in {
                    p.lower() for p in dedicated_parts
                }:
                    dedicated_parts.append(part_norm)
            if dedicated_parts:
                dedicated = " ".join(dedicated_parts)
                if dedicated not in queries:
                    queries.append(dedicated)
                    query_locales.append("en")

        # Append ONE cuisine keyword per query (cycling) for cultural dietary styles.
        if dietary_style and dietary_style.value in _CUISINE_SIGNATURES:
            keywords = _CUISINE_SIGNATURES[dietary_style.value]
            enriched: list[str] = []
            for idx, q in enumerate(queries):
                enriched.append(f"{q} {keywords[idx % len(keywords)]}")
            queries = enriched

        return queries

    @staticmethod
    def _time_to_max_ingredients(available_minutes: Optional[int]) -> Optional[int]:
        """
        Converts the user's available cooking time into a max_ingredients ceiling.
        """
        if available_minutes is None:
            return None
        if available_minutes <= 15:
            return 4
        if available_minutes <= 20:
            return 6
        if available_minutes <= 30:
            return 8
        if available_minutes <= 60:
            return 10
        if available_minutes <= 90:
            return 12
        return 15

    def _gather_recipe_candidates(
        self,
        approved_assessments: list[IngredientAssessment],
        dietary_style: Optional[DietaryStyle] = None,
        dish_type_filter: Optional[str] = None,
        max_ingredients: Optional[int] = None,
        priority_barcode: Optional[str] = None,
    ) -> list[dict]:
        """
        Two-stage retrieval pipeline (no LLM involved):
          1. Combinatoric role-aware over-fetching — up to 5 queries, each fetching
             _RAW_FETCH_PER_QUERY results, deduplicated by title.
             Optional ChromaDB metadata pre-filters (dish_type, ingredient_count)
             are applied at the database level before semantic ranking.
          2. Coverage ranking — sort the entire pool by ingredient availability
             ratio (best-covered first); return the top _MAX_CANDIDATES.

        There is NO hard coverage cutoff — the best available candidates always
        reach the LLM, even when the user's pantry is unusual or sparse.
        """
        queries = self._build_search_queries(
            approved_assessments, dietary_style, priority_barcode
        )
        priority_assessment = self._find_priority_assessment(
            approved_assessments, priority_barcode
        )
        priority_tokens = (
            _priority_tokens(priority_assessment) if priority_assessment else set()
        )

        # Use generic_name_en for coverage scoring — these are the same clean English terms
        # used in the queries, maximising token-overlap with recipe ingredient lists.
        approved_names = [
            a.substitution_suggestion or a.generic_name_en
            for a in approved_assessments
        ]

        seen_titles: set[str] = set()
        raw_pool: list[dict] = []

        def _merge_batch(batch: list[dict]) -> None:
            for recipe in batch:
                title = recipe["title"]
                if title not in seen_titles:
                    seen_titles.add(title)
                    raw_pool.append(recipe)

        filter_info = (
            f" [dish={dish_type_filter or 'any'}, max_ing={max_ingredients or '∞'}]"
        )
        nq = len(queries)
        workers = min(4, nq) if nq else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for query in queries:
                print(f"[Chef] Searching recipes for: '{query}'{filter_info}")
                futures.append(
                    executor.submit(
                        self._tool.fetch_candidates,
                        query,
                        _RAW_FETCH_PER_QUERY,
                        dish_type_filter,
                        max_ingredients,
                    )
                )
            for fut in futures:
                _merge_batch(fut.result())

        if priority_assessment and priority_tokens:
            has_priority_in_pool = any(
                _recipe_dict_has_priority(r, priority_tokens) for r in raw_pool
            )
            if not has_priority_in_pool:
                pin_en = _culinary_keyword_en(priority_assessment)
                pin_es = _culinary_keyword_es(priority_assessment)
                extra_q = " ".join(
                    p for p in dict.fromkeys([pin_en, pin_es]) if p
                )
                print(
                    f"[Chef] No priority matches in pool; extra search: '{extra_q}'"
                )
                extra_batch = self._tool.fetch_candidates(
                    extra_q,
                    _RAW_FETCH_PER_QUERY,
                    dish_type_filter,
                    max_ingredients,
                )
                _merge_batch(extra_batch)

        print(f"[Chef] Raw pool: {len(raw_pool)} unique candidates.")

        # min_anchor: require at least 2 scanned ingredients to appear in a TIER-0 recipe
        # (or all of them, if the user scanned fewer than 2 approved items).
        min_anchor = min(2, len(approved_names))
        ranked = self._rank_by_coverage(
            raw_pool,
            approved_names,
            min_anchor,
            priority_barcode=priority_barcode,
            approved_assessments=approved_assessments,
        )

        top = ranked[: self._MAX_CANDIDATES]
        if priority_tokens:
            in_top = sum(1 for r in top if _recipe_dict_has_priority(r, priority_tokens))
            if in_top < 2:
                with_priority = [
                    r for r in ranked if _recipe_dict_has_priority(r, priority_tokens)
                ]
                without_priority = [
                    r for r in ranked if not _recipe_dict_has_priority(r, priority_tokens)
                ]
                if with_priority:
                    top = (with_priority + without_priority)[: self._MAX_CANDIDATES]
                    print(
                        f"[Chef] Boosted {len(with_priority)} priority candidate(s) "
                        f"into top-{self._MAX_CANDIDATES}."
                    )
        return top

    @staticmethod
    def _rank_by_coverage(
        candidates: list[dict],
        approved_names: list[str],
        min_anchor: int,
        priority_barcode: Optional[str] = None,
        approved_assessments: Optional[list[IngredientAssessment]] = None,
    ) -> list[dict]:
        """
        Two-tier coverage ranking with DUAL entry conditions for TIER 0.

        A recipe is TIER 0 (high-priority, sent to LLM first) only when it passes BOTH:
          A) Anchor Rule:    at least `min_anchor` of its ingredients match the user's
                             SCANNED items (approved_names).  _BASE_PANTRY items do NOT
                             count here — the recipe must genuinely use the photo pantry.
          B) Coverage Rule:  (scanned + base pantry) / total ingredients ≥ 0.75.

        A recipe that scores on base-pantry items alone (oil, salt, garlic) but ignores
        the user's scanned products will fail Condition A and land in TIER 1.

        No recipe is ever discarded — TIER 1 candidates still reach the LLM so the
        system can always propose 3 options even when the knowledge base has no exact match.
        Within each tier, results are sorted by total coverage ratio descending.
        """
        user_tokens: set[str] = {name.lower().strip() for name in approved_names}

        priority_tokens: set[str] = set()
        if priority_barcode and approved_assessments:
            pinned = ChefAgent._find_priority_assessment(
                approved_assessments, priority_barcode
            )
            if pinned is not None:
                priority_tokens = _priority_tokens(pinned)

        def _matches(item_name: str, token_set: set[str]) -> bool:
            return any(
                len(token) >= 3 and (token in item_name or item_name in token)
                for token in token_set
            )

        # scored: (tier, priority_sort_key, user_count, total_ratio, recipe)
        scored: list[tuple[int, int, int, float, dict]] = []

        for recipe in candidates:
            items = _parse_stored_list(recipe.get("ingredients", ""))
            if not items:
                # No ingredient data — treat as fully covered, passes both conditions
                scored.append((0, 0, min_anchor, 1.0, recipe))
                continue

            user_covered  = 0
            total_covered = 0
            for item in items:
                item_name = (
                    item.get("ingredient") or item.get("name") or ""
                ).lower().strip()
                if not item_name:
                    continue
                in_user = _matches(item_name, user_tokens)
                in_base = _matches(item_name, _BASE_PANTRY)
                if in_user:
                    user_covered  += 1
                if in_user or in_base:
                    total_covered += 1

            total       = len(items)
            total_ratio = total_covered / total if total > 0 else 1.0

            # Check if the recipe contains the pinned priority ingredient
            has_priority_item = True
            priority_sort_key = 0
            if priority_tokens:
                combined_recipe_text = (
                    f"{recipe.get('title', '')} {recipe.get('ingredients', '')}"
                )
                has_priority_item = _text_has_priority_token(
                    combined_recipe_text, priority_tokens
                )
                priority_sort_key = 0 if has_priority_item else 1

            # Dual gate: both conditions must hold for TIER 0
            anchor_ok   = user_covered >= min_anchor   # Condition A
            coverage_ok = total_ratio  >= 0.75         # Condition B
            tier = 0 if (anchor_ok and coverage_ok and has_priority_item) else 1
            scored.append((tier, priority_sort_key, user_covered, total_ratio, recipe))

        # Multi-key sort — in order of priority:
        #   1. Tier ascending  (TIER 0 before TIER 1)
        #   2. Priority sort key ascending (Recipes containing the pinned item first)
        #   3. Total coverage ratio descending  (penalise missing ingredients)
        #   4. Absolute user-scanned ingredient count descending
        scored.sort(key=lambda x: (x[0], x[1], -x[3], -x[2]))

        tier0_count = sum(1 for s in scored if s[0] == 0)
        best_user   = max((s[2] for s in scored), default=0)
        best_total  = max((s[3] for s in scored), default=0.0)
        print(
            f"[Chef] Coverage ranking — {len(candidates)} candidates | "
            f"tier-0 (anchor≥{min_anchor} + coverage≥75%): {tier0_count} | "
            f"best user-anchor: {best_user} items | best total: {best_total:.0%}."
        )
        return [recipe for _, _, _, _, recipe in scored]

    @staticmethod
    def _format_priority_block(assessment: IngredientAssessment) -> str:
        en = _culinary_keyword_en(assessment)
        es = _culinary_keyword_es(assessment)
        return (
            f"## USER PRIORITY INGREDIENT (MANDATORY)\n"
            f"The user pinned this ingredient — it MUST appear structurally in every recipe's "
            f"`available_ingredients` (scaled grams, not optional garnish):\n"
            f"  English culinary name : {en or '(see Spanish)'}\n"
            f"  Spanish culinary name : {es or '(see English)'}\n"
            f"  Barcode               : {assessment.barcode}\n"
            f"  Caloric density       : {assessment.kcal_per_100g} kcal/100g\n\n"
        )

    @staticmethod
    def _ensure_priority_in_proposal(
        proposal: ChefProposal,
        priority_assessment: IngredientAssessment,
        diners: int,
        output_language: Literal["es", "en"],
    ) -> ChefProposal:
        """Inject pinned ingredient into recipe 1 if the LLM omitted it from all proposals."""
        tokens = _priority_tokens(priority_assessment)
        display_name = (
            _culinary_keyword_es(priority_assessment)
            if output_language == "es"
            else _culinary_keyword_en(priority_assessment)
        )
        if not display_name:
            display_name = (
                priority_assessment.generic_name_en
                or priority_assessment.generic_name_es
                or "priority ingredient"
            )

        def _recipe_has_priority(recipe: RecipeProposal) -> bool:
            for ing in recipe.available_ingredients:
                if _text_has_priority_token(ing.name, tokens):
                    return True
            return False

        if any(_recipe_has_priority(r) for r in proposal.recipes):
            return proposal

        role = priority_assessment.culinary_role
        if role == CulinaryRole.MAIN_PROTEIN:
            grams = 80 * diners
        elif role == CulinaryRole.MAIN_CARB:
            grams = 50 * diners
        elif role == CulinaryRole.PRODUCE:
            grams = 60 * diners
        else:
            grams = 30 * diners

        target = proposal.recipes[0]
        target.available_ingredients = list(target.available_ingredients) + [
            RecipeIngredient(name=display_name, quantity_g=grams)
        ]
        if output_language == "es":
            step = (
                f"Incorpora {grams} g de {display_name} según los pasos de la receta "
                f"(ingrediente prioritario del usuario)."
            )
        else:
            step = (
                f"Incorporate {grams} g of {display_name} according to the recipe steps "
                f"(user-pinned priority ingredient)."
            )
        target.preparation_steps = [step, *list(target.preparation_steps)]
        print(
            "[Chef] Priority ingredient missing from LLM output; "
            f"injected into recipe 1 ({display_name}, {grams} g)."
        )
        return proposal

    def _build_user_message(
        self,
        report: NutritionistReport,
        diners: int,
        approved_assessments: list[IngredientAssessment],
        excluded: list[str],
        candidates_text: str,
        dietary_style: Optional[DietaryStyle] = None,
        output_language: Literal["es", "en"] = "es",
        priority_assessment: Optional[IngredientAssessment] = None,
    ) -> str:
        """
        Assembles the complete human-turn message with all contextual data the
        LLM needs: nutritional constraints, ingredient availability (with caloric
        density for accurate STEP 5 calculation), dietary preference, and the
        recipe candidate pool.
        """
        approved_block = "\n".join(
            f" {a.substitution_suggestion or a.generic_name_en} "
            f"(Density: {a.kcal_per_100g} kcal/100g)"
            for a in approved_assessments
        ) or "  (none)"
        excluded_block = "\n".join(f"  • {ing}" for ing in excluded) or "  (none)"

        pantry_block = ", ".join(_BASE_PANTRY_DISPLAY)
        diet_line = dietary_style.value if dietary_style else "None declared"

        if output_language == "es":
            output_lang_block = (
                "## OUTPUT LANGUAGE\n"
                "OUTPUT LANGUAGE: es\n"
                "Escribe los títulos de receta, nombres de ingredientes, "
                "`preparation_steps` y `justification` completamente en español.\n\n"
            )
        else:
            output_lang_block = (
                "## OUTPUT LANGUAGE\n"
                "OUTPUT LANGUAGE: en\n"
                "Write recipe titles, ingredient names, `preparation_steps`, and "
                "`justification` entirely in English.\n\n"
            )

        priority_block = (
            self._format_priority_block(priority_assessment)
            if priority_assessment is not None
            else ""
        )

        return (
            f"{output_lang_block}"
            f"{priority_block}"
            f"## BASE CONSTRAINTS (from Nutritionist Agent)\n"
            f"Meal Caloric Target  : {report.meal_caloric_target} kcal per serving\n"
            f"Daily Caloric Target : {report.daily_caloric_target} kcal\n"
            f"Daily Protein Target : {report.daily_protein_g} g\n"
            f"Daily Carb Target    : {report.daily_carb_g} g\n"
            f"Daily Fat Target     : {report.daily_fat_g} g\n"
            f"\n## DIETARY PREFERENCE\n"
            f"  {diet_line}\n"
            f"\n## DINERS\n"
            f"Recipes must be scaled for: {diners} person(s)\n"
            f"\n## BASE PANTRY (assumed available for every user — treat as available_ingredients)\n"
            f"  {pantry_block}\n"
            f"\n## APPROVED INGREDIENTS (user physically has these — use them)\n"
            f"{approved_block}\n"
            f"\n## EXCLUDED INGREDIENTS (allergens/dietary violations — NEVER include in any recipe)\n"
            f"{excluded_block}\n"
            f"\n## CANDIDATE RECIPES (pre-ranked by ingredient coverage — best match first)\n\n"
            f"{candidates_text}"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, request: ChefRequest) -> ChefProposal:
        """
        Executes the full Chef pipeline and returns a structured proposal.

        Args:
            request: ChefRequest containing the Nutritionist report and diner count.

        Returns:
            ChefProposal with exactly 3 adapted, scaled, and calorie-verified recipes.
        """
        print(f"[Chef] Starting recipe proposal for {request.diners} diner(s)...")

        # Step 1: extract ingredient constraint lists from the nutritionist report
        approved_assessments, excluded = self._extract_ingredient_lists(
            request.nutritionist_report
        )
        print(
            f"[Chef] Constraints — "
            f"approved: {len(approved_assessments)} ingredient(s) | "
            f"excluded: {len(excluded)} allergen(s)"
        )

        # Translate available cooking time into a max_ingredients complexity ceiling
        max_ingredients = self._time_to_max_ingredients(request.available_minutes)
        if max_ingredients is not None:
            print(
                f"[Chef] Time constraint: {request.available_minutes} min "
                f"→ max {max_ingredients} ingredients per recipe."
            )

        # Step 2: retrieve and rank recipe candidates (combinatoric queries + coverage ranking)
        candidates = self._gather_recipe_candidates(
            approved_assessments,
            dietary_style=request.dietary_style,
            dish_type_filter=request.dish_type_filter,
            max_ingredients=max_ingredients,
            priority_barcode=request.priority_barcode,
        )
        if not candidates:
            raise RuntimeError(
                "[Chef] No recipe candidates found in the knowledge base. "
                "Ensure the vector store is populated before running the Chef Agent."
            )

        priority_assessment = self._find_priority_assessment(
            approved_assessments, request.priority_barcode
        )

        # Step 3: format candidates and build the LLM prompt
        candidates_text = _format_candidates_as_text(candidates)
        priority_selector_block = (
            self._format_priority_block(priority_assessment)
            if priority_assessment is not None
            else ""
        )
        selector_message = (
            f"{priority_selector_block}"
            "## CANDIDATE RECIPES (ranked best-first)\n\n"
            f"{candidates_text}\n\n"
            "Select exactly 3 candidate indices."
        )
        print("[Chef] Invoking LLM selector...")
        selection: ChefSelection = self._selector_chain.invoke({"user_message": selector_message})

        # Build a smaller pool for the full constructor: only the 3 selected candidates.
        chosen: list[dict] = []
        for idx in selection.selected_indexes:
            if 1 <= idx <= len(candidates):
                chosen.append(candidates[idx - 1])
        # Fallback if something went wrong: take top-3
        if len(chosen) != 3:
            chosen = candidates[:3]

        chosen_text = _format_candidates_as_text(chosen)
        user_message = self._build_user_message(
            request.nutritionist_report,
            request.diners,
            approved_assessments,
            excluded,
            chosen_text,
            request.dietary_style,
            request.output_language,
            priority_assessment=priority_assessment,
        )

        print("[Chef] Invoking LLM for recipe adaptation and proposal (selected candidates only)...")
        proposal: ChefProposal = self._chain.invoke({"user_message": user_message})

        if priority_assessment is not None:
            proposal = self._ensure_priority_in_proposal(
                proposal,
                priority_assessment,
                request.diners,
                request.output_language,
            )

        print(f"[Chef] Proposal generated: {len(proposal.recipes)} recipe(s).")
        return proposal


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from src.agents.nutritionist import (
        IngredientAssessment,
        NutritionistReport,
    )

    # Minimal synthetic NutritionistReport to avoid needing a live OpenFoodFacts call.
    # Simulates a user with: wheat flour (approved), olive oil (approved),
    # eggs (approved), milk (substitute_required → lactose-free milk),
    # and crustaceans (excluded allergen — e.g. shrimp).
    # kcal_per_100g values are realistic figures sourced from standard nutritional tables.
    from src.agents.nutritionist import CulinaryRole  # noqa: F811 (re-import for clarity)
    mock_report = NutritionistReport(
        daily_caloric_target=2480,
        meal_caloric_target=827,
        daily_protein_g=186,
        daily_carb_g=279,
        daily_fat_g=69,
        ingredient_assessment=[
            IngredientAssessment(
                barcode="0000000001",
                product_name="Harina de trigo integral extra fina",
                status=IngredientStatus.APPROVED,
                reason="No allergen conflict detected.",
                kcal_per_100g=364.0,
                culinary_role=CulinaryRole.MAIN_CARB,
                generic_name_en="flour",
                generic_name_es="harina",
            ),
            IngredientAssessment(
                barcode="0000000002",
                product_name="Aceite de oliva virgen extra premium",
                status=IngredientStatus.APPROVED,
                reason="No allergen conflict detected.",
                kcal_per_100g=884.0,
                culinary_role=CulinaryRole.BASE_CONDIMENT,
                generic_name_en="olive oil",
                generic_name_es="aceite de oliva",
            ),
            IngredientAssessment(
                barcode="0000000003",
                product_name="Huevos frescos camperos talla M",
                status=IngredientStatus.APPROVED,
                reason="No allergen conflict detected.",
                kcal_per_100g=155.0,
                culinary_role=CulinaryRole.MAIN_PROTEIN,
                generic_name_en="eggs",
                generic_name_es="huevos",
            ),
            IngredientAssessment(
                barcode="0000000004",
                product_name="Leche entera de vaca pasteurizada",
                status=IngredientStatus.SUBSTITUTE_REQUIRED,
                reason="Contains lactose; user has lactose intolerance.",
                substitution_suggestion="lactose-free milk",
                kcal_per_100g=42.0,
                culinary_role=CulinaryRole.BASE_CONDIMENT,
                generic_name_en="milk",
                generic_name_es="leche",
            ),
            IngredientAssessment(
                barcode="0000000005",
                product_name="Gambas cocidas peladas congeladas",
                status=IngredientStatus.EXCLUDED,
                reason="User has a severe crustacean allergy.",
                kcal_per_100g=99.0,
                culinary_role=CulinaryRole.MAIN_PROTEIN,
                generic_name_en="shrimp",
                generic_name_es="gambas",
            ),
        ],
    )

    request = ChefRequest(
        nutritionist_report=mock_report,
        diners=2,
        dietary_style=DietaryStyle.MEDITERRANEAN,
    )

    print("\n" + "=" * 60)
    print("CHEF AGENT — TEST RUN")
    print("=" * 60)

    agent = ChefAgent()
    proposal = agent.run(request)

    print("\n--- STRUCTURED CHEF PROPOSAL ---")
    print(json.dumps(proposal.model_dump(), indent=2, ensure_ascii=False))
