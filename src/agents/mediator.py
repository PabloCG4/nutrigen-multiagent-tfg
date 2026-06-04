from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load the project .env before any LangChain / OpenAI client is instantiated,
# so OPENAI_API_KEY is available from the moment the module is imported.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

from src.agents.chef import ChefProposal, RecipeIngredient, RecipeProposal
from src.agents.nutritionist import (
    MacroBreakdown,
    NutritionistCritique,
    RecipeCritique,
)
from src.agents import config


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Mediator Agent — the final "Integrator" in the multi-agent \
culinary assistant pipeline.

You receive:
  1. USER CONTEXT       — time available to cook, cheat-meal flag, free-text special requests.
  2. CHEF PROPOSAL      — 3 fully adapted, scaled, and calorie-estimated recipes (with
                          available and missing vital ingredients already in grams).
  3. NUTRITIONIST CRITIQUE — macro balance verdict and optional correction suggestions per recipe.

Your mission: for ALL 3 recipes, apply any Nutritionist corrections, evaluate and sutil-adapt viable special requests, integrate new ingredients into the preparation steps, and produce 3 complete, polished, cook-ready FinalRecipe outputs.
You are an INTEGRATOR, not a selector — your output MUST contain all 3 recipes.
Complete the following four steps in order.

────────────────────────────────────────────
STEP 1 — Estimate Preparation Time
────────────────────────────────────────────
For EACH of the 3 recipes, estimate the total preparation time in minutes using:
  • Number and complexity of preparation steps.
  • Ingredient types (raw meat, hard vegetables, dried legumes that need soaking, etc.).
  • Techniques explicitly mentioned in the steps (e.g., "simmer 20 min", "bake 35 min",
    "marinate overnight" → long; "toss and serve" → short).

────────────────────────────────────────────
STEP 2 — Evaluate Each Recipe Against User Context and Special Requests
────────────────────────────────────────────
For EACH recipe, assess its fit and capture the result in the recipe's `justification` field:

  a) TIME FIT:
       If `available_minutes` is provided:
         — note whether the recipe fits within the limit or exceeds it, and by how much.
       If `available_minutes` is null:
         — note whether it is quick (<20 min), moderate (20–50 min), or long (>50 min).

  b) CHEAT MEAL:
       If `is_cheat_meal = true`:
         — note that nutritional constraints are advisory; flavour is prioritised.
       If `is_cheat_meal = false`:
         — note the Nutritionist's macro verdict and whether any deficit is concerning.

  c) SPECIAL REQUESTS VIABILITY GATEKEEPER:
       Analyze the user's free-text `special_requests` against the current recipe structure:
         • VIABLE REQUESTS: Subtle culinary modifications that can be solved exclusively by altering cooking techniques or utilizing universal staples from the BASE PANTRY (e.g., "less spicy", "more spicy", "crunchy textures").
         • NON-VIABLE REQUESTS: Demands that require deleting structural ingredients validated by the Nutritionist, adding complex missing proteins/carbohydrates not present in the inventory, or violating allergy restrictions.
       If a request is non-viable or clashes with the dish type (e.g., asking for a crunchy texture in a smooth pureed cream), you MUST discard the request for this specific recipe and explicitly document the technical reason inside the `justification` field.

────────────────────────────────────────────
STEP 3 — Integrate Corrections and Subtle Culinary Adaptations (for EACH of the 3 recipes)
────────────────────────────────────────────
For EACH recipe, build a single, unified set of cooking instructions by integrating both nutritional fixes and viable user preferences:

  A) Nutritional Corrections:
       If `correction_suggestions` from the Nutritionist is NON-EMPTY:
         1. Add each suggested ingredient to `ingredients` with source = "nutritionist_addition".
            Assign a sensible quantity in GRAMS scaled for the diner count.
         2. Rewrite `preparation_steps` to integrate the new ingredient at the culinarily correct moment.

  B) Special Requests Adaptation (Only for viable requests verified in STEP 2):
       Re-write specific lines of the `preparation_steps` to accommodate subtle preferences without changing the ingredient quantities or macro balances:
         • For "crunchy textures" / "texturas crujientes": Modify cooking techniques instructions. Adjust vegetable steps to be sautéed briefly ("leave vegetables al dente to preserve a firm, crunchy texture") or instruct to toast base pantry bread or existing starches with olive oil until crisp.
         • For "less spicy" / "menos picante": Modify instructions to reduce or make optional any inherent hot spices or pepper mentioned in the original steps.
         • For "more spicy" / "más picante": Add an explicit step instructing to season the dish generously with black pepper from the BASE PANTRY during cooking or plating.

  C) Verbatim Fallback:
       If `correction_suggestions` is empty and no special requests are evaluated as viable for this recipe, copy the Chef's original `preparation_steps` verbatim. Do NOT alter them.

────────────────────────────────────────────
STEP 4 — Build Unified Ingredient Lists (for EACH of the 3 recipes)
────────────────────────────────────────────
For each recipe, combine all ingredients into one flat list using three source tags:
  • source = "available"            → from `available_ingredients` (user already has these)
  • source = "to_buy"                 → from `missing_ingredients`   (user must purchase these)
  • source = "nutritionist_addition"  → ingredients added in STEP 3  (new, with estimated grams)

────────────────────────────────────────────
OUTPUT RULES (NON-NEGOTIABLE)
────────────────────────────────────────────
  • `recipes` MUST contain EXACTLY 3 entries, in the same order as the Chef's proposals.
  • ALL ingredient quantities as integers in grams.
  • `estimated_time_minutes` — positive integer, in minutes.
  • `preparation_steps` — fully integrated version (rewritten if corrections or viable subtle culinary adaptations were applied, verbatim from Chef otherwise).
  • `justification` — for each recipe: (a) estimated time and fit vs user constraint, (b) Nutritionist's verdict and whether corrections were applied, (c) explicit alignment notes detailing how the viable special requests were fulfilled, OR a clear professional explanation justifying why the request was discarded as non-viable for this specific dish.
  • `caloric_note` — estimated kcal per individual serving and a one-sentence macro balance summary after any Nutritionist corrections.

The user's human message includes an OUTPUT LANGUAGE block. All user-visible text in
every FinalRecipe (titles, ingredient names, preparation_steps, justification,
caloric_note) MUST follow that language — do not mix languages.
"""


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class UserContext(BaseModel):
    available_minutes: Optional[int] = Field(
        default=None,
        ge=5,
        le=240,
        description=(
            "Maximum minutes the user can spend cooking. "
            "None means no time constraint (agent defaults to moderate 30–50 min recipes)."
        )
    )
    is_cheat_meal: bool = Field(
        default=False,
        description=(
            "When True, the user wants an indulgent meal; "
            "the Nutritionist's macro warnings are treated as advisory only."
        )
    )
    special_requests: Optional[str] = Field(
        default=None,
        description=(
            "Free-text user preferences: flavour profile, cuisine style, "
            "dish type, or any other context (e.g., 'something spicy', 'light lunch')."
        )
    )
    output_language: Literal["es", "en"] = Field(
        default="es",
        description="Language for all user-facing recipe text (titles, steps, notes).",
    )


class MediatorRequest(BaseModel):
    user_context: UserContext = Field(
        description="Dynamic session context provided by the user before cooking"
    )
    chef_proposal: ChefProposal = Field(
        description="The 3 adapted recipe proposals produced by the Chef Agent"
    )
    nutritionist_critique: NutritionistCritique = Field(
        description="The per-recipe macro audit produced by the Nutritionist Agent (Phase 2)"
    )
    diners: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of people to serve; used to assign quantities for correction ingredients"
    )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class IngredientSource(str, Enum):
    AVAILABLE             = "available"
    TO_BUY                = "to_buy"
    NUTRITIONIST_ADDITION = "nutritionist_addition"


class FinalIngredient(BaseModel):
    name: str = Field(
        description="Ingredient name"
    )
    quantity_g: int = Field(
        description="Quantity in grams (scaled for the full diner count)"
    )
    source: IngredientSource = Field(
        description=(
            "'available' = user already has it; "
            "'to_buy' = structurally vital, must be purchased; "
            "'nutritionist_addition' = added by the Nutritionist to fix a macro deficit"
        )
    )


class FinalRecipe(BaseModel):
    title: str = Field(
        description="Title of the recipe, matching the Chef's proposal exactly"
    )
    estimated_time_minutes: int = Field(
        ge=1,
        description="Estimated total preparation + cooking time in minutes"
    )
    justification: str = Field(
        description=(
            "Time estimate and fit vs user constraint, Nutritionist's macro verdict, "
            "whether corrections were applied or the recipe was already balanced, "
            "and alignment with the user's special requests."
        )
    )
    ingredients: List[FinalIngredient] = Field(
        description=(
            "Unified ingredient list from all three sources "
            "(available, to_buy, nutritionist_addition). All quantities in grams."
        )
    )
    preparation_steps: List[str] = Field(
        description=(
            "Definitive, fully integrated cooking steps. "
            "Rewritten to include Nutritionist corrections if any were suggested; "
            "otherwise verbatim from the Chef."
        )
    )
    caloric_note: str = Field(
        description=(
            "Estimated kcal per individual serving and a one-sentence macro balance "
            "summary reflecting any Nutritionist corrections applied."
        )
    )
    macro_breakdown: Optional[MacroBreakdown] = Field(
        default=None,
        description=(
            "Exact per-serving macro totals computed by Phase 3 (Nutritionist). "
            "Populated after the Mediator stage; None until Phase 3 completes."
        )
    )

    @field_validator("ingredients")
    @classmethod
    def ingredients_must_not_be_empty(cls, v: List[FinalIngredient]) -> List[FinalIngredient]:
        """The final recipe must have at least one ingredient."""
        if not v:
            raise ValueError("FinalRecipe.ingredients must contain at least one ingredient.")
        return v

    @field_validator("preparation_steps")
    @classmethod
    def steps_must_not_be_empty(cls, v: List[str]) -> List[str]:
        """The final recipe must have at least one preparation step."""
        if not v:
            raise ValueError("FinalRecipe.preparation_steps must contain at least one step.")
        return v


class MediatorOutput(BaseModel):
    """
    The terminal output of the full multi-agent pipeline.
    Contains all 3 recipes, each fully corrected and display-ready.
    """
    recipes: List[FinalRecipe] = Field(
        description=(
            "All 3 corrected and integrated recipes, in the same order as the "
            "Chef's proposals. Each has its Nutritionist corrections applied."
        )
    )

    @field_validator("recipes")
    @classmethod
    def must_be_exactly_three(cls, v: List[FinalRecipe]) -> List[FinalRecipe]:
        """Enforces exactly 3 output recipes — one per Chef proposal."""
        if len(v) != 3:
            raise ValueError(
                f"MediatorOutput must contain exactly 3 recipes, got {len(v)}."
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# THE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class MediatorAgent:
    """
    Mediator Agent: the final "Integrator" in the multi-agent culinary pipeline.

    Workflow:
      1. Formats a unified LLM prompt containing the user's session context, all 3 Chef
         recipe proposals, and the Nutritionist's per-recipe macro critiques.
      2. Invokes the LLM with structured output to:
           a. Estimate preparation time for each recipe.
           b. Evaluate each recipe against the user's time constraint, cheat-meal flag,
              and special requests (result goes into each recipe's justification).
           c. Integrate Nutritionist correction suggestions into preparation steps
              (rewrite steps if corrections exist; copy verbatim otherwise).
           d. Build unified ingredient lists (available / to_buy / nutritionist_addition).
           e. Produce 3 complete FinalRecipe objects, one per Chef proposal.
      3. Returns a MediatorOutput containing all 3 corrected recipes.

    This agent holds no tools — all its reasoning is done in a single LLM call,
    with every input pre-structured by the upstream agents.
    """

    def __init__(self, model: str = config.MEDIATOR_LLM_MODEL) -> None:
        print("--- Initializing Mediator Agent ---")

        llm = ChatOpenAI(model=model, temperature=0)

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "{user_message}"),
        ])
        self._chain = self._prompt | llm.with_structured_output(MediatorOutput)

        print("--- Mediator Agent Ready ---")

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_recipe_block(
        recipe: RecipeProposal,
        critique: RecipeCritique,
        index: int,
    ) -> str:
        """
        Renders one recipe and its paired critique side-by-side into a readable block.
        Keeping them together reduces cross-referencing errors in the LLM.
        """
        avail = "\n".join(
            f"        - {ing.name}: {ing.quantity_g} g"
            for ing in recipe.available_ingredients
        ) or "        (none)"

        missing = "\n".join(
            f"        - {ing.name}: {ing.quantity_g} g"
            for ing in recipe.missing_ingredients
        ) or "        (none)"

        # Trim steps to reduce prompt size; mediator can rewrite when needed.
        steps = "\n".join(
            f"        {i}. {step}"
            for i, step in enumerate(recipe.preparation_steps[:8], start=1)
        )
        if len(recipe.preparation_steps) > 8:
            steps += "\n        ... (trimmed)"

        corrections = (
            "\n".join(
                f"        - {s.ingredient_name}: {s.reason}"
                for s in critique.correction_suggestions
            )
            if critique.correction_suggestions
            else "        (none — recipe is nutritionally balanced)"
        )

        return (
            f"RECIPE {index}: {recipe.title}\n"
            f"Estimated calories per serving : {recipe.estimated_calories_per_serving} kcal\n"
            f"Available ingredients:\n{avail}\n"
            f"Missing ingredients:\n{missing}\n"
            f"Preparation steps (trimmed):\n{steps}\n"
            f"Nutritionist verdict : {critique.macro_verdict.value}\n"
            f"Deficiencies         : {critique.detected_deficiencies}\n"
            f"Correction suggestions:\n{corrections}\n"
            f"---\n"
        )

    def _build_user_message(self, request: MediatorRequest) -> str:
        """
        Assembles the complete human-turn message with:
          - User session context.
          - Each of the 3 recipes paired with its Nutritionist critique.
          - Nutritionist's overall assessment summary.
        """
        ctx = request.user_context
        time_line = (
            f"{ctx.available_minutes} minutes"
            if ctx.available_minutes is not None
            else "not specified (assume moderate duration)"
        )
        cheat_line = (
            "YES — prioritise flavour and satisfaction; nutritional rules are advisory"
            if ctx.is_cheat_meal
            else "NO — respect Nutritionist verdicts fully"
        )
        requests_line = ctx.special_requests or "none"

        if ctx.output_language == "es":
            output_lang_block = (
                "## OUTPUT LANGUAGE\n"
                "OUTPUT LANGUAGE: es\n"
                "Todo el texto visible al usuario en cada FinalRecipe (títulos, nombres de "
                "ingredientes, `preparation_steps`, `justification`, `caloric_note`) debe "
                "estar en español.\n\n"
            )
        else:
            output_lang_block = (
                "## OUTPUT LANGUAGE\n"
                "OUTPUT LANGUAGE: en\n"
                "All user-visible text in each FinalRecipe (titles, ingredient names, "
                "`preparation_steps`, `justification`, `caloric_note`) must be in English.\n\n"
            )

        # Pair each recipe with its matching critique by position
        recipe_blocks = ""
        critiques = request.nutritionist_critique.recipe_critiques
        for i, (recipe, critique) in enumerate(
            zip(request.chef_proposal.recipes, critiques), start=1
        ):
            recipe_blocks += self._format_recipe_block(recipe, critique, i)

        return (
            f"{output_lang_block}"
            f"## USER CONTEXT\n"
            f"Available cooking time : {time_line}\n"
            f"Cheat meal             : {cheat_line}\n"
            f"Special requests       : {requests_line}\n"
            f"Diners                 : {request.diners} person(s)\n"
            f"\n## NUTRITIONIST ASSESSMENT SUMMARY\n"
            f"{request.nutritionist_critique.overall_recommendation}\n"
            f"\n## RECIPE PROPOSALS + NUTRITIONIST CRITIQUES\n\n"
            f"{recipe_blocks.rstrip()}"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, request: MediatorRequest) -> MediatorOutput:
        """
        Executes the full mediation pipeline and returns all 3 corrected recipes.

        Args:
            request: MediatorRequest bundling user context, Chef proposals,
                     and Nutritionist critiques.

        Returns:
            MediatorOutput — 3 fully integrated, corrected, display-ready recipes.
        """
        print("[Mediator] Starting integration of all 3 recipes...")

        user_message = self._build_user_message(request)

        print("[Mediator] Invoking LLM for recipe integration and correction...")
        output: MediatorOutput = self._chain.invoke({"user_message": user_message})

        titles = [r.title for r in output.recipes]
        print(f"[Mediator] Integration complete. Recipes: {titles}")
        return output


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from src.agents.nutritionist import (
        CorrectionSuggestion,
        MacroVerdict,
        NutritionistCritique,
        RecipeCritique,
    )
    from src.agents.chef import (
        ChefProposal,
        RecipeIngredient,
        RecipeProposal,
    )

    # ── Synthetic Chef Proposal ───────────────────────────────────────────────
    # Recipe 1: Pasta al Tomate    — carb-heavy, protein_deficit expected
    # Recipe 2: Tortilla Española  — balanced (eggs, potato, olive oil)
    # Recipe 3: Crema de Verduras  — light soup, calorie_deficit expected
    mock_chef_proposal = ChefProposal(
        recipes=[
            RecipeProposal(
                title="Pasta al Tomate con Aceite de Oliva",
                justification="High carb overlap with approved ingredients.",
                estimated_calories_per_serving=480,
                available_ingredients=[
                    RecipeIngredient(name="wheat flour pasta", quantity_g=150),
                    RecipeIngredient(name="tomato sauce",      quantity_g=120),
                    RecipeIngredient(name="olive oil",         quantity_g=20),
                ],
                missing_ingredients=[],
                preparation_steps=[
                    "Bring a large pot of salted water to a boil.",
                    "Cook pasta until al dente (≈10 min); drain.",
                    "Warm tomato sauce in a pan over medium heat.",
                    "Combine pasta and sauce; drizzle with olive oil and serve.",
                ],
            ),
            RecipeProposal(
                title="Tortilla Española",
                justification="Strong macro coverage; eggs and potato cover protein and carbs.",
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
                    "Peel and thinly slice potatoes and onion.",
                    "Fry in olive oil over low heat until soft (≈15 min); drain excess oil.",
                    "Beat eggs with salt; fold in the potato-onion mixture.",
                    "Pour into a non-stick pan; cook on both sides until set (≈5 min each).",
                    "Slide onto a plate and let rest 2 minutes before slicing.",
                ],
            ),
            RecipeProposal(
                title="Crema de Verduras",
                justification="Light vegetable soup; low calorie, high micronutrient profile.",
                estimated_calories_per_serving=180,
                available_ingredients=[
                    RecipeIngredient(name="olive oil", quantity_g=15),
                ],
                missing_ingredients=[
                    RecipeIngredient(name="zucchini",        quantity_g=200),
                    RecipeIngredient(name="carrot",          quantity_g=100),
                    RecipeIngredient(name="onion",           quantity_g=80),
                    RecipeIngredient(name="vegetable broth", quantity_g=400),
                ],
                preparation_steps=[
                    "Dice zucchini, carrot, and onion.",
                    "Sauté onion in olive oil over medium heat until translucent (≈3 min).",
                    "Add zucchini and carrot; cook 5 minutes.",
                    "Pour in vegetable broth; simmer until vegetables are tender (≈20 min).",
                    "Blend until smooth; season with salt and serve.",
                ],
            ),
        ]
    )

    # ── Synthetic Nutritionist Critique ───────────────────────────────────────
    mock_critique = NutritionistCritique(
        recipe_critiques=[
            RecipeCritique(
                recipe_title="Pasta al Tomate con Aceite de Oliva",
                macro_verdict=MacroVerdict.PROTEIN_DEFICIT,
                detected_deficiencies=(
                    "Protein ≈ 8 g vs target 62 g per meal — severe deficit. "
                    "Carbs and fat are adequate."
                ),
                correction_suggestions=[
                    CorrectionSuggestion(
                        ingredient_name="canned tuna",
                        reason="Lean, neutral-flavoured protein source; pairs well with tomato pasta.",
                    ),
                ],
            ),
            RecipeCritique(
                recipe_title="Tortilla Española",
                macro_verdict=MacroVerdict.BALANCED,
                detected_deficiencies="No significant macro deficiencies detected.",
                correction_suggestions=[],
            ),
            RecipeCritique(
                recipe_title="Crema de Verduras",
                macro_verdict=MacroVerdict.CALORIE_DEFICIT,
                detected_deficiencies=(
                    "Estimated 180 kcal vs meal target 827 kcal — well below 60% threshold."
                ),
                correction_suggestions=[
                    CorrectionSuggestion(
                        ingredient_name="cooked chickpeas",
                        reason="Adds protein and complex carbs; classic in vegetable soups.",
                    ),
                    CorrectionSuggestion(
                        ingredient_name="crusty bread",
                        reason="Significant calorie boost; traditional accompaniment to creamy soups.",
                    ),
                ],
            ),
        ],
        overall_recommendation=(
            "Tortilla Española is the most balanced option: all macros within range, "
            "moderate cooking time, and high ingredient availability. "
            "Pasta al Tomate needs tuna for protein; Crema de Verduras needs calorie boosting."
        ),
    )

    user_ctx = UserContext(
        available_minutes=30,
        is_cheat_meal=False,
        special_requests="Something filling and easy to make on a weeknight.",
    )

    mock_request = MediatorRequest(
        user_context=user_ctx,
        chef_proposal=mock_chef_proposal,
        nutritionist_critique=mock_critique,
        diners=2,
    )

    print("\n" + "=" * 60)
    print("MEDIATOR AGENT — TEST RUN (3-recipe integration)")
    print("=" * 60)

    agent = MediatorAgent()
    output = agent.run(mock_request)

    print("\n--- MEDIATOR OUTPUT (3 recipes) ---")
    print(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))
