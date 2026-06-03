"""
Adds two metadata columns to every row:
ingredient_count  (int):number of ingredients parsed from `final_ingredients`.
dish_type         (str): macro-category inferred from the recipe title using a Title-First-Match heuristic over exhaustive keyword lists.

"""

import ast
import csv
import os
import re
from typing import Optional

import pandas as pd

from src.text_engine import config

ENRICHED_RECIPE_PATH: str = os.path.join(
    config.MASTER_DATA_DIR,
    "master_recipe_dataset_enriched.csv",
)

DISH_TYPE_KEYWORDS: dict[str, list[str]] = {

    "soup_stew": [
        # English: generic
        "soup", "soupe", "broth", "bisque", "chowder", "consomme", "consommé",
        "bouillabaisse", "potage", "velouté", "veloute", "bouillon",
        # English: stews and braises
        "stew", "braise", "braised", "ragù", "ragu", "ragout", "fricassee",
        "goulash", "cassoulet", "pot roast", "pot-roast",
        # English: specific soups
        "minestrone", "ramen", "pho", "laksa", "miso", "tom yum", "tom kha",
        "gazpacho", "vichyssoise", "mulligatawny", "borscht", "hotpot",
        "hot pot", "pozole", "menudo", "caldo", "caldillo",
        # English: chilis and curries (liquid-based)
        "chili", "chilli",
        # Spanish: generic
        "sopa", "caldo", "crema", "puré", "pure", "potaje", "puchero",
        "cocido", "guiso", "estofado", "escudella", "olla",
        # Spanish: specific dishes
        "lentejas", "judías", "judias", "alubias", "garbanzos", "fabada",
        "gazpacho", "ajoblanco", "salmorejo", "pisto", "ropa vieja",
        "sancocho", "ajiaco", "mondongo", "locro", "sancochado",
        "cazuela", "caldereta", "marmitako", "berberechos",
        # Italian: other European
        "zuppa", "ribollita", "acquacotta", "brodetto",
        # Asian
        "congee", "juk", "sundubu", "doenjang",
    ],

    "salad_side": [
        # English: salads
        "salad", "slaw", "coleslaw", "ceviche", "poke", "tabbouleh",
        "fattoush", "waldorf", "nicoise", "niçoise", "caprese",
        "insalata", "ensalada", "salade",
        # English: side dishes and vegetables
        "side", "sides", "side dish", "garnish", "accompaniment",
        "roasted vegetables", "grilled vegetables", "steamed vegetables",
        "stir-fry", "stir fry", "sauté", "saute", "sautéed", "sauteed",
        "glazed carrots", "mashed", "mash", "puree", "purée",
        "au gratin", "gratin", "pickled", "pickle", "kimchi",
        "vinaigrette", "dressing", "relish",
        # English: grain and legume sides
        "pilaf", "pilau", "pilav", "couscous", "tabbouleh", "hummus",
        "refried beans", "rice pilaf",
        # Spanish — salads and sides
        "ensalada", "ensaladilla", "remojón", "remojón",
        "escalivada", "panaché", "panache", "guarnición", "guarnicion",
        "verduras", "menestra", "pisto", "samfaina",
        # Specific well-known sides
        "guacamole", "tzatziki", "raita", "chutney", "salsa",
        "baba ganoush", "baba ghanoush", "muhammara", "ezme",
    ],

    "dessert": [
        # English: cakes and pastries
        "cake", "cupcake", "cheesecake", "pound cake", "sponge cake",
        "layer cake", "bundt", "loaf", "gâteau", "gateau",
        "pastry", "pastries", "strudel", "croissant", "danish",
        "tart", "tartlet", "pie", "cobbler", "crisp", "crumble",
        "clafoutis", "galette", "quiche",  # quiche is savoury but often sweet-listed
        # English: cookies and biscuits
        "cookie", "cookies", "biscuit", "biscuits", "shortbread",
        "brownie", "blondie", "bar", "bars", "square", "squares",
        "fudge", "truffle", "truffles",
        # English: puddings and creams
        "pudding", "mousse", "custard", "crème brûlée", "creme brulee",
        "panna cotta", "pannacotta", "tiramisu", "tiramisù",
        "affogato", "soufflé", "souffle", "parfait", "trifle",
        "bread pudding",
        # English: frozen
        "ice cream", "gelato", "sorbet", "sherbet", "semifreddo",
        "frozen yogurt", "frozen yoghurt", "popsicle",
        # English: doughnuts and fried sweets
        "donut", "doughnut", "churros", "beignet", "funnel cake",
        "fritter", "zeppole", "loukoumades",
        # English: confectionery
        "candy", "caramel", "praline", "nougat", "marzipan",
        "meringue", "macaroon", "macaron",
        # Spanish: cakes and pastries
        "tarta", "bizcocho", "magdalena", "muffin", "pastel",
        "brazo de gitano", "buñuelo", "buñuelos",
        "churros", "porras", "pestiños", "roscón", "roscon",
        "ensaimada", "coca", "polvorón", "polvorones", "mantecado",
        "turrón", "turron",
        # Spanish: puddings and creams
        "flan", "crema catalana", "natillas", "arroz con leche",
        "leche frita", "torrija", "torrijas", "bizcochos borrachos",
        "mousse de chocolate", "tarta de queso", "postre",
        # Spanish: frozen
        "helado", "sorbete",
        # Spanish: cookies and small sweets
        "galleta", "galletas", "alfajor", "alfajores", "suspiro",
        # Latin American
        "tres leches", "dulce de leche", "suspiro limeño",
        "tembleque", "tembleque",
        # Other: international
        "baklava", "halva", "kheer", "gulab jamun", "ladoo",
        "mochi", "daifuku", "dorayaki", "dorayaki",
        "profiterole", "eclair", "éclair", "religieuse",
        "opera cake", "schwarzwälder", "schwarzwalder",
    ],

    "snack_breakfast": [
        # English: breakfast items
        "breakfast", "brunch", "oatmeal", "porridge", "granola",
        "muesli", "cereal", "overnight oats", "french toast",
        "pancake", "pancakes", "waffle", "waffles", "crepe", "crêpe",
        "crêpes", "crepes", "frittata", "omelette", "omelet",
        "scrambled eggs", "fried eggs", "poached eggs", "eggs benedict",
        "shakshuka", "huevos rancheros",
        # English: snacks and finger food
        "snack", "snacks", "appetizer", "appetisers", "appetizers",
        "starter", "starters", "canape", "canapé", "canapés",
        "finger food", "tapas", "antipasto", "antipasti",
        "bruschetta", "crostini", "toast", "toastie",
        "cracker", "crackers", "chip", "chips", "nachos",
        "popcorn", "trail mix",
        # English: dips and spreads
        "dip", "dips", "spread", "spreads",
        "pâté", "pate", "rillettes",
        # English: sandwiches and wraps (considered snack/light)
        "sandwich", "sandwiches", "wrap", "wraps", "roll", "rolls",
        "sub", "hoagie", "baguette", "pita", "flatbread",
        "quesadilla", "taco", "tacos", "burrito", "burritos",
        "empanada", "empanadas", "arepa", "arepas",
        # English: drinks and smoothies
        "smoothie", "smoothies", "shake", "milkshake",
        "juice", "lemonade", "cocktail", "mocktail",
        "infusion", "tea", "latte", "coffee drink",
        # Spanish: breakfast
        "desayuno", "almuerzo ligero", "tostada", "tostadas",
        "pan con tomate", "pa amb tomàquet",
        "gachas", "migas", "porridge de avena",
        # Spanish: snacks and tapas
        "tapa", "tapas", "aperitivo", "aperitivos", "snack",
        "pinchos", "pintxos", "montadito", "montaditos",
        "croqueta", "croquetas", "patatas bravas", "pa amb oli",
        "boquerón", "boquerones", "anchoa", "anchoas",
        "albóndiga", "albondigas",  # can be tapas
        "banderilla", "banderillas", "tigre", "tigres",
        # Spanish: bocadillos and wraps
        "bocadillo", "bocadillos", "bocata", "sándwich", "sandwich",
        "wrap", "enrollado",
        # Latin American snacks
        "pupusa", "pupusas", "tamale", "tamales", "gordita", "gorditas",
        "chalupa", "chalupas", "tostada", "tostadas", "chilaquiles",
        "enchilada", "enchiladas",
        # Other international snacks
        "samosa", "samosas", "pakora", "pakoras", "spring roll",
        "spring rolls", "gyoza", "dumpling", "dumplings",
        "dim sum", "shumai", "har gow", "baozi", "bao",
        "onigiri", "temaki",
    ],

    "drink": [
        # English
        "smoothie", "smoothies", "shake", "milkshake", "milkshakes",
        "juice", "lemonade", "limeade", "agua fresca",
        "cocktail", "mocktail", "punch",
        "sangria", "sangría", "tinto de verano",
        "horchata", "atole", "champurrado",
        "kefir", "lassi",
        # Spanish
        "batido", "zumo", "refresco", "granizado", "limonada",
        "bebida", "cóctel", "coctel",
    ],
}

# Flat ordered list of categories for the Title-First-Match heuristic.
CATEGORY_ORDER: list[str] = [
    "drink",
    "dessert",
    "snack_breakfast",
    "soup_stew",
    "salad_side",
]

# Pre-compile one regex pattern per category for performance.
# Keywords are sorted longest-first to maximise match specificity.
COMPILED_PATTERNS: list[tuple[str, re.Pattern]] = []
for cat in CATEGORY_ORDER:
    keywords = sorted(DISH_TYPE_KEYWORDS[cat], key=len, reverse=True)
    # Use word-boundary anchors so "pie" does not match "spice".
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
        flags=re.IGNORECASE,
    )
    COMPILED_PATTERNS.append((cat, pattern))

def _parse_ingredient_list(raw_value: object) -> list:
    """
    Safely parses a `final_ingredients` cell into a Python list.
    """
    if isinstance(raw_value, list):
        return raw_value
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []
    try:
        parsed = ast.literal_eval(raw_value.strip())
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _classify_dish_type(title: str) -> str:
    """
    Applies the Title-First-Match heuristic to infer a dish macro-category.

    Iterates over the pre-compiled keyword patterns in CATEGORY_ORDER and
    returns the first category whose pattern matches anywhere in the title.
    Returns "main_course" when no pattern matches.
    """
    lowered = title.lower() if isinstance(title, str) else ""
    for category, pattern in COMPILED_PATTERNS:
        if pattern.search(lowered):
            return category
    return "main_course"

def enrich_dataset() -> None:
    """
    Reads `master_recipe_dataset.csv`, adds `ingredient_count` and `dish_type`
    columns, and writes the result to `master_recipe_dataset_enriched.csv`.
    """
    if not os.path.exists(config.MASTER_RECIPE_PATH):
        raise FileNotFoundError(
            f"Master dataset not found at: {config.MASTER_RECIPE_PATH}\n"
        )

    print(f"Reading master dataset from:\n  {config.MASTER_RECIPE_PATH}")
    df: pd.DataFrame = pd.read_csv(config.MASTER_RECIPE_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
    print(f"Columns: {list(df.columns)}")

    print("\nCalculating ingredient_count …")
    df["ingredient_count"] = df["final_ingredients"].apply(
        lambda cell: len(_parse_ingredient_list(cell))
    )
    zero_count = (df["ingredient_count"] == 0).sum()
    if zero_count:
        print(
            f"  Warning: {zero_count:,} rows have ingredient_count=0 "
        )
    print(f"  ingredient_count stats:\n{df['ingredient_count'].describe()}")

    print("\nClassifying dish_type …")
    df["dish_type"] = df["title"].apply(_classify_dish_type)
    distribution = df["dish_type"].value_counts()
    print(f"  dish_type distribution:\n{distribution.to_string()}")

    assert os.path.abspath(ENRICHED_RECIPE_PATH) != os.path.abspath(
        config.MASTER_RECIPE_PATH
    ), "SAFETY ABORT: output path resolves to the same file as the source."

    # Write enriched CSV
    os.makedirs(config.MASTER_DATA_DIR, exist_ok=True)
    print(f"\nSaving enriched dataset to:\n  {ENRICHED_RECIPE_PATH}")
    df.to_csv(
        ENRICHED_RECIPE_PATH,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )
    print(
        f"Done. Enriched dataset saved with {len(df):,} rows "
        f"and {len(df.columns)} columns."
    )
    print(f"New columns added: ingredient_count, dish_type")

if __name__ == "__main__":
    enrich_dataset()
