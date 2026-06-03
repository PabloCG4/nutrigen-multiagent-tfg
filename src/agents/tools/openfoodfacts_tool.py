from typing import Type, Optional, Tuple
import openfoodfacts
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# Module-level initialization of the official API client
# This automatically handles the User-Agent and prevents HTTP 429 Too Many Requests
off_api = openfoodfacts.API(user_agent="TFG-PersonalChef-Project/1.0")

VITAL_MACRO_KEYS: Tuple[str, ...] = ("energy-kcal", "proteins", "carbohydrates", "fat")

class OpenFoodFactsInput(BaseModel):
    barcode: str = Field(
        description="The EAN/UPC barcode of the product to look up (e.g., '8410076472480')."
    )

class OpenFoodFactsTool(BaseTool):
    """
    Tool that allows the Nutritionist Agent to retrieve nutritional data
    for a product by querying the OpenFoodFacts public API with its barcode.

    If the barcode lookup returns a product with no macronutrient data, a
    secondary fallback search is attempted automatically using the product's
    most specific category tag, ensuring reliable macronutrient retrieval.

    Used only when the local database is not available (products_test_database_macros.json)
    """
    name: str = "lookup_nutritional_data"
    description: str = (
        "Use this tool to retrieve nutritional information for a food product "
        "given its EAN or UPC barcode. Returns product name, calories, "
        "macronutrients (proteins, carbohydrates, fats per 100g), and allergens. "
        "Input must be a single barcode string."
    )
    args_schema: Type[BaseModel] = OpenFoodFactsInput

    def fetch_category_fallback_nutriments(self, category_tag: str) -> Optional[dict]:
        """
        Performs a category-based search on OpenFoodFacts and returns the nutriments
        dict of the first result that contains all vital macronutrient values.
        Returns None if the search fails or no suitable product is found.
        """
        # Clean the category tag to make a robust text search via the SDK
        clean_category = category_tag.replace("en:", "").replace("-", " ")
        try:
            search_results = off_api.product.text_search(clean_category)
            products = search_results.get("products", [])
        except Exception as error:
            print(f" Fallback API error for category '{clean_category}': {error}")
            return None

        for candidate in products:
            candidate_nutriments = candidate.get("nutriments", {})
            if has_all_vital_macros(candidate_nutriments):
                return candidate_nutriments

        print(f" No suitable fallback product found with complete macros for category '{clean_category}'.")
        return None

    def _run(self, barcode: str) -> str:
        """
        Queries the OpenFoodFacts API and returns a structured nutritional
        profile formatted for LLM consumption.
        """
        try:
            # SDK query handles HTTP errors internally and returns a dictionary
            product_data = off_api.product.get(barcode)
        except Exception as error:
            return (
                f" Error when querying barcode '{barcode}' using OpenFoodFacts SDK: {error}. "
                "No nutritional data available."
            )

        if not product_data:
            return (
                f" Barcode '{barcode}' was not found in the OpenFoodFacts "
                "database. The product may not be catalogued yet."
            )

        # The SDK might return the dictionary unwrapped or wrapped inside "product"
        actual_product = product_data.get("product", product_data)
        nutriments = actual_product.get("nutriments", {})

        # Prioritize English generic names, then local generic, then English commercial, then local commercial
        product_name = (
            actual_product.get("product_name")
            or actual_product.get("product_name_es")
            or "Unknown Product"
        )
        brands = actual_product.get("brands", "Unknown")
        quantity = actual_product.get("quantity", "Unknown")

        categories_hierarchy = actual_product.get("categories_hierarchy", [])
        best_category = (
            categories_hierarchy[-1].replace("en:", "").replace("-", " ") 
            if categories_hierarchy 
            else "Unknown category"
        )

        categories_tags = actual_product.get("categories_tags", [])
        macro_note: str = ""
        
        if are_vital_macros_missing(nutriments) and categories_tags:
            specific_category = categories_tags[-1]
            fallback_nutriments = self.fetch_category_fallback_nutriments(specific_category)
            if fallback_nutriments is not None:
                nutriments = fallback_nutriments
                clean_category_name = specific_category.replace("en:", "").replace("-", " ")
                macro_note = f"Note: Macros estimated from a similar product in the same category ({clean_category_name})."

        calories     = get_nutriment(nutriments, "energy-kcal")
        proteins     = get_nutriment(nutriments, "proteins")
        carbs        = get_nutriment(nutriments, "carbohydrates")
        fats         = get_nutriment(nutriments, "fat")
        fiber        = get_nutriment(nutriments, "fiber")
        sugars       = get_nutriment(nutriments, "sugars")
        salt         = get_nutriment(nutriments, "salt")

        allergens_raw = actual_product.get("allergens", "") or actual_product.get("allergens_tags", [])
        if isinstance(allergens_raw, list):
            allergens = ", ".join(
                tag.replace("en:", "").replace("-", " ").title()
                for tag in allergens_raw
            ) or "None declared"
        else:
            allergens = allergens_raw.strip() or "None declared"

        nova_group   = actual_product.get("nova_group", "Unknown")
        nutriscore   = actual_product.get("nutriscore_grade", "Unknown").upper()

        report = (
            f"=== NUTRITIONAL PROFILE ===\n"
            f"Barcode        : {barcode}\n"
            f"Product Name   : {product_name}\n"
            f"Brand          : {brands}\n"
            f"Category       : {best_category}\n"
            f"Package Size   : {quantity}\n"
            f"\n-- Macronutrients (per 100 g) --\n"
            f"Calories       : {calories} kcal\n"
            f"Proteins       : {proteins} g\n"
            f"Carbohydrates  : {carbs} g\n"
            f"  of which Sugars: {sugars} g\n"
            f"Fats           : {fats} g\n"
            f"Dietary Fiber  : {fiber} g\n"
            f"Salt           : {salt} g\n"
            f"\n-- Health Indicators --\n"
            f"Nutri-Score    : {nutriscore}\n"
            f"NOVA Group     : {nova_group} (1=unprocessed, 4=ultra-processed)\n"
            f"\n-- Allergens --\n"
            f"Allergens      : {allergens}\n"
        )
        if macro_note:
            report += f"\n[!] {macro_note}\n"

        report += "==========================="
        return report

    async def _arun(self, barcode: str) -> str:
        """
        Async compatibility shim delegates to the synchronous implementation.
        """
        return self._run(barcode)

def are_vital_macros_missing(nutriments: dict) -> bool:
    """
    Returns True if ALL four vital macronutrient fields are absent or zero,
    indicating that the product entry lacks meaningful nutritional data.
    """
    return all(
        get_nutriment(nutriments, key) == "N/A"
        for key in VITAL_MACRO_KEYS
    )

def has_all_vital_macros(nutriments: dict) -> bool:
    """
    Returns True only if all four vital macronutrient fields are present.
    """
    return all(
        get_nutriment(nutriments, key) != "N/A"
        for key in VITAL_MACRO_KEYS
    )

def get_nutriment(nutriments: dict, key: str) -> str:
    """
    Returns the per-100g value for a nutriment key, falling back to the
    plain key, then to 'N/A' if neither is present.
    """
    value = nutriments.get(f"{key}_100g") or nutriments.get(key)
    if value is None or value == "":
        return "N/A"
    try:
        return str(round(float(value), 2))
    except ValueError:
        return "N/A"

if __name__ == "__main__":
    tool = OpenFoodFactsTool()

    test_cases = [
        ("00004717", "Aceite de oliva (likely missing macros -> triggers fallback)"),
        ("00000000000000", "Non-existent barcode"),
        ("00014423", "Galletas de soda (should have macros -> no fallback)"),
    ]

    for test_barcode, label in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {label}  |  Barcode: {test_barcode}")
        print('='*60)
        result = tool.invoke({"barcode": test_barcode})
        print(result)