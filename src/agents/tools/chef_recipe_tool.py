from __future__ import annotations

import ast
import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool

from src.text_engine.rag.retrieval_service import RecipeRetriever, get_recipe_retriever


# --- INPUT SCHEMA (reuses the same single-field pattern as RecipeSearchTool) ---
class ChefRecipeSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Semantic search query for recipes based on available ingredients "
            "(e.g., 'chicken tomato pasta', 'arroz pollo verduras')."
        )
    )


# --- THE CHEF'S RECIPE SEARCH TOOL ---
class ChefRecipeSearchTool(BaseTool):
    """
    Extended recipe search tool for the Chef Agent.

    Unlike the general-purpose RecipeSearchTool, this tool queries ChromaDB
    directly to retrieve FULL recipe metadata — including preparation instructions —
    which RecipeRetriever._refine_ranking() discards in its standard return value.
    It also exposes a fetch_candidates() method for programmatic use by the Chef
    Agent (deterministic retrieval before the LLM reasoning phase).
    """

    name: str = "search_recipes_for_chef"
    description: str = (
        "Searches the recipe knowledge base and returns full recipe data including "
        "ingredient lists (with quantities and units) and preparation instructions. "
        "Use ingredient names as the search query."
    )
    args_schema: Type[BaseModel] = ChefRecipeSearchInput

    # Internal retriever — protected from Pydantic validation via PrivateAttr
    _retriever: RecipeRetriever = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        print("Initializing Chef Recipe Search Tool")
        self._retriever = get_recipe_retriever()

    def fetch_candidates(
        self,
        query: str,
        top_k: int = 3,
        dish_type: Optional[str] = None,
        max_ingredients: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB directly to retrieve full recipe metadata, including
        preparation instructions that RecipeRetriever._refine_ranking() omits.

        Applies deterministic metadata pre-filtering BEFORE semantic ranking so
        that the candidate pool is already constrained by dish category and recipe
        complexity.  Applies the same NLTK-based re-ranking as RecipeRetriever so
        that ingredient-keyword overlap is rewarded, then returns the top_k results.

        Args:
            query:           Semantic search string (ingredient names or dish description).
            top_k:           Maximum number of recipes to return.
            dish_type:       Optional ChromaDB `dish_type` value to filter on
                             (e.g. "soup_stew", "dessert", "main_course").
                             Skipped when None or empty string.
            max_ingredients: Optional upper bound for `ingredient_count` metadata field.
                             Recipes with more ingredients than this value are excluded.
                             Skipped when None.

        Returns:
            List of dicts with keys: title, ingredients, instructions, source, url.
        """
        t0 = time.perf_counter()
        query_vector = self._retriever.model.encode(query).tolist()
        t_encode = time.perf_counter()

        # Build the ChromaDB `where` clause from the active filters.
        # ChromaDB requires `$and` when more than one condition is present.
        where: Optional[Dict[str, Any]] = None
        conditions: List[Dict[str, Any]] = []
        if dish_type:
            conditions.append({"dish_type": {"$eq": dish_type}})
        if max_ingredients is not None:
            conditions.append({"ingredient_count": {"$lte": max_ingredients}})

        if len(conditions) == 2:
            where = {"$and": conditions}
        elif len(conditions) == 1:
            where = conditions[0]
        # len == 0 → where stays None (no pre-filter)

        # NOTE: Avoid per-query `collection.count()` calls (can be expensive on large persistent stores).
        # We intentionally over-fetch a small constant factor and re-rank deterministically.
        fetch_n = max(top_k * 2, top_k)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results":        fetch_n,
            "include":          ["metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        raw = self._retriever.collection.query(**query_kwargs)
        t_query = time.perf_counter()

        if not raw["ids"] or not raw["ids"][0]:
            t_end = time.perf_counter()
            print(
                f"[ChefRecipeSearchTool][Timing] encode={(t_encode - t0):.3f}s | "
                f"query={(t_query - t_encode):.3f}s | rerank=0.000s | total={(t_end - t0):.3f}s"
            )
            return []

        # Re-rank using the same keyword-boost logic as RecipeRetriever,
        # but preserve 'instructions' which _refine_ranking normally discards
        ranked = self._rerank_with_instructions(query, raw, top_k)
        t_rerank = time.perf_counter()
        print(
            f"[ChefRecipeSearchTool][Timing] encode={(t_encode - t0):.3f}s | "
            f"query={(t_query - t_encode):.3f}s | rerank={(t_rerank - t_query):.3f}s | total={(t_rerank - t0):.3f}s"
        )
        return ranked

    def _run(self, query: str) -> str:
        """
        Returns a formatted string of recipe candidates for direct LLM consumption.
        Delegates retrieval to fetch_candidates() and formats the results.
        """
        try:
            candidates = self.fetch_candidates(query, top_k=5)
            if not candidates:
                return f"No recipes found in the knowledge base for query: '{query}'."
            return _format_candidates_as_text(candidates)
        except Exception as e:
            return f"Error during recipe search: {e}"

    async def _arun(self, query: str) -> str:
        """Async compatibility shim — delegates to the synchronous implementation."""
        return self._run(query)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _rerank_with_instructions(
        self, query: str, raw: dict, final_k: int
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks raw ChromaDB results using the same NLTK keyword-boost logic
        as RecipeRetriever._refine_ranking(), while also extracting instructions
        from the metadata (which the standard implementation discards).
        """
        query_lower = query.lower()
        stop_words = self._retriever.stop_words

        important_words = [w for w in query_lower.split() if w not in stop_words]
        if not important_words:
            important_words = query_lower.split()

        scored: List[Dict[str, Any]] = []
        for meta, dist in zip(raw["metadatas"][0], raw["distances"][0]):
            ingredients = meta.get("ingredients", "").lower()
            score       = dist

            # Keyword overlap boosts
            matches_ingr  = sum(1 for w in important_words if w in ingredients)
            if matches_ingr > 0:
                score *= 0.95 ** matches_ingr

            scored.append({
                "title":        meta.get("title", "Unknown"),
                "ingredients":  meta.get("ingredients", ""),
                "instructions": meta.get("instructions", ""),
                "source":       meta.get("source", "Unknown"),
                "url":          meta.get("url", ""),
                "_score":       score,
            })

        scored.sort(key=lambda x: x["_score"])

        # Strip internal scoring field before returning
        results = []
        for item in scored[:final_k]:
            results.append({k: v for k, v in item.items() if k != "_score"})
        return results


def _parse_stored_list(raw_string: str) -> list:
    """
    Safely parses a string representation of a Python list stored in ChromaDB
    (e.g. "[{'ingredient': 'flour', 'quantity': 2.0, 'unit': 'cups'}]").
    Returns an empty list if parsing fails.
    """
    if not raw_string or raw_string in ("", "[]", "nan"):
        return []
    try:
        result = ast.literal_eval(raw_string)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def _format_candidates_as_text(candidates: List[Dict[str, Any]]) -> str:
    """Renders a list of candidate recipe dicts into an LLM-readable string block."""
    sections = []
    for i, recipe in enumerate(candidates, 1):
        # Trim to keep prompts small (latency). Chef only needs enough to pick/adapt.
        ingredients_raw = recipe.get("ingredients", "")
        instructions_raw = recipe.get("instructions", "")

        ingredients_items = _parse_stored_list(ingredients_raw) if isinstance(ingredients_raw, str) else []
        ingredients_preview = ""
        if ingredients_items:
            preview_lines = []
            for item in ingredients_items[:12]:
                name = item.get("ingredient") or item.get("name") or "unknown"
                preview_lines.append(f"  - {name}")
            ingredients_preview = "\n".join(preview_lines)
        else:
            ingredients_preview = "  (ingredient data unavailable)"

        steps_items = _parse_stored_list(instructions_raw) if isinstance(instructions_raw, str) else []
        if steps_items:
            steps_preview = "\n".join(f"  {j}. {step}" for j, step in enumerate(steps_items[:6], 1))
        else:
            steps_preview = "  (preparation steps unavailable in database)"

        block = (
            f"--- CANDIDATE RECIPE {i} ---\n"
            f"Title  : {recipe['title']}\n"
            f"\nIngredients (names only, trimmed):\n"
            f"{ingredients_preview}\n"
            f"\nPreparation Steps (trimmed):\n"
            f"{steps_preview}"
        )
        sections.append(block)
    return "\n\n".join(sections)


# TEST BLOCK
if __name__ == "__main__":
    tool = ChefRecipeSearchTool()

    test_query = "pasta con tomate y atún"
    print(f"\nSearching for: '{test_query}'\n{'='*60}")

    # Test the programmatic API (returns structured data)
    candidates = tool.fetch_candidates(test_query, top_k=3)
    print(f"Fetched {len(candidates)} candidates.\n")
    for c in candidates:
        print(f"  Title: {c['title']}")
        print(f"  Has instructions: {'yes' if c['instructions'] else 'no'}\n")

    # Test the LangChain tool interface (returns formatted string)
    result = tool.invoke({"query": test_query})
    print("\nFORMATTED OUTPUT FOR LLM:\n")
    print(result[:1000], "..." if len(result) > 1000 else "")
