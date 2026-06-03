import pandas as pd
import ast
import re
from fractions import Fraction
import os
from src.text_engine import config

# --- Text Processing Definitions ---

# List of valid measurement units to be identified within ingredient strings
VALID_UNITS = [
    'c', 'cup', 'cups', 'oz', 'ounce', 'ounces', 'pkg', 'package',
    'tsp', 'tsps', 'teaspoon', 'teaspoons',
    'Tbsp', 'tbsp', 'Tbsps', 'tablespoon', 'tablespoons',
    'g', 'gram', 'grams', 'kg', 'kilo', 'kilos',
    'lb', 'lbs', 'pound', 'pounds',
    'ml', 'milliliter', 'milliliters', 'l', 'liter', 'liters',
    "taza", "tazas", "litro", "onza", "onzas", "paquete", "paquetes",
    "cucharada", "cucharadas", "cucharadita", "cucharaditas",
    "gramos", "kilogramos", "mililitros", "litros"
]

# Pre-compiled regular expressions for performance optimization during bulk processing
UNITS_PATTERN = r'\b(' + r'|'.join(VALID_UNITS) + r')\b\.?'
RE_PAREN_UNIT = re.compile(r'\((?P<q>\d+\.?\d*)\s*(?P<u>' + '|'.join(VALID_UNITS) + r')\)', re.IGNORECASE)
RE_RANGE = re.compile(r'^(?P<q1>\d+\.?\d*)\s*(?:to|-)\s*(?P<q2>\d+\.?\d*)', re.IGNORECASE)
RE_QUANTITY = re.compile(r'^(?P<quantity>\d+\s*\d*/\d*|\d+/\d+|\d+\.?\d*)', re.IGNORECASE)
RE_UNIT = re.compile(r'^' + UNITS_PATTERN, re.IGNORECASE)

# --- Helper Functions ---

def convert_quantity_to_float(quantity_string: str) -> float:
    """
    Converts text-based quantities (fractions, decimals, or mixed numbers) into floats.
    Example: "1 1/2" becomes 1.5.
    """
    if not quantity_string:
        return 1.0
    try:
        # Split mixed numbers like "1 1/2" and sum their fractional values
        string_parts = quantity_string.split()
        return float(sum(Fraction(s) for s in string_parts))
    except (ValueError, ZeroDivisionError):
        # Fallback to 1.0 if the format is unrecognized
        return 1.0

def parse_single_ingredient(ingredient_string: str) -> dict:
    """
    Parses raw ingredient strings into structured data.
    Implements fallback logic to prevent 'Unknown' names when parsing fails.
    """
    raw_input = str(ingredient_string).strip()
    if not raw_input:
        return {"name": "Unknown", "quantity": 1.0, "unit": "unit"}

    current_string = raw_input
    quantity_str, unit = "1", "unit"
    found_quantity = False

    # 1. Handle parentheses: (500 g) flour
    paren_match = RE_PAREN_UNIT.search(current_string)
    if paren_match:
        quantity_str, unit = paren_match.group('q'), paren_match.group('u')
        current_string = RE_PAREN_UNIT.sub('', current_string).strip()
        found_quantity = True
    
    # 2. Handle numeric ranges: 1-2 cups (calculates the average)
    if not found_quantity:
        range_match = RE_RANGE.match(current_string)
        if range_match:
            v1, v2 = float(range_match.group('q1')), float(range_match.group('q2'))
            quantity_str = str((v1 + v2) / 2.0)
            current_string = current_string[range_match.end():].strip()
            found_quantity = True
    
    # 3. Handle standard leading quantities (e.g., '2 cloves' or '1scoop')
    if not found_quantity:
        qty_match = RE_QUANTITY.match(current_string)
        if qty_match:
            quantity_str = qty_match.group('quantity').strip()
            current_string = current_string[qty_match.end():].strip()

    # 4. Extract unit if present
    unit_match = RE_UNIT.match(current_string)
    if unit_match:
        unit = unit_match.group(0).replace('.', '').strip()
        current_string = current_string[unit_match.end():].strip()

    # --- FALLBACK LOGIC ---
    # Clean the remaining name
    final_name = current_string.lstrip(',').strip()
    
    # If parsing logic left the name empty or too short, use the original raw input
    if not final_name or len(final_name) < 2:
        final_name = raw_input

    return {
        "name": final_name,
        "quantity": convert_quantity_to_float(quantity_str),
        "unit": unit.lower()
    }

def convert_string_to_list(text_data):
    """Safely converts string representations of lists into actual Python list objects."""
    if isinstance(text_data, str):
        try:
            return ast.literal_eval(text_data)
        except (ValueError, SyntaxError):
            return []
    return []

# --- Dataset Specific Processing Logic ---

def reconcile_nlg_row(row):
    """
    Cross-references raw ingredient strings with Named Entity Recognition (NER) 
    tags to create a structured list of ingredients.
    """
    try:
        raw_ingredients = ast.literal_eval(row['ingredients'])
        ner_names = ast.literal_eval(row['NER'])
        
        parsed_ingredients = [parse_single_ingredient(item) for item in raw_ingredients]
        
        # If lengths mismatch, default to the parsed names from the raw string
        if len(parsed_ingredients) != len(ner_names):
            return [
                {"ingredient": p["name"], "quantity": p["quantity"], "unit": p["unit"]} 
                for p in parsed_ingredients
            ]
        
        # Merge NER clean names with quantities parsed from raw strings
        structured_list = []
        for clean_name, parsed_data in zip(ner_names, parsed_ingredients):
            structured_list.append({
                "ingredient": clean_name,
                "quantity": parsed_data["quantity"],
                "unit": parsed_data["unit"]
            })
        return structured_list
    except (ValueError, SyntaxError, KeyError):
        return []

def process_recipenlg_dataset():
    """Processes the large-scale RecipeNLG CSV dataset."""
    # Use path from config
    input_path = config.RECIPENLG_RAW_PATH
    output_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_RecipeNLG.csv")

    # Access filename safely using os.path.basename
    print(f"\n--- Processing Large Dataset: {os.path.basename(input_path)} ---")
    if not os.path.exists(input_path):
        print(f"Error: Dataset not found at {input_path}")
        return None

    # Processing a sample of 1000 rows for development testing
    # Use pd.read_csv(input_path) for the full run
    recipe_df = pd.read_csv(input_path).head(1000)
    
    # Generate structured ingredient lists
    recipe_df['final_ingredients'] = recipe_df.apply(reconcile_nlg_row, axis=1)
    
    # Parse instruction strings into list format
    recipe_df['structured_directions'] = recipe_df['directions'].apply(convert_string_to_list)
    
    # --- FINAL CLEANUP ---
    # Ensure all required columns exist and filter out the rest (NER, raw ingredients, etc.)
    if 'link' not in recipe_df.columns:
        recipe_df['link'] = "RecipeNLG"
    if 'source' not in recipe_df.columns:
        recipe_df['source'] = "RecipeNLG"
        
    final_df = recipe_df[config.OUTPUT_COLUMNS]
    
    final_df.to_csv(output_path, index=False)
    return final_df

def process_scraped_json_file(filename):
    """Processes JSON files generated from web scraping scripts."""
    # Use paths from config
    input_path = os.path.join(config.RAW_DATA_DIR, filename)
    output_path = os.path.join(config.PROCESSED_DATA_DIR, f"processed_{filename.replace('.json', '.csv')}")

    print(f"\n--- Processing Scraped File: {filename} ---")
    if not os.path.exists(input_path):
        print(f"Skipping: {filename} not found.")
        return None

    scraped_df = pd.read_json(input_path)
    
    # Filter out empty records without using abstract map(bool) logic
    scraped_df = scraped_df[scraped_df['ingredients'].str.len() > 0]
    scraped_df = scraped_df[scraped_df['directions'].str.len() > 0].copy()

    # Define internal logic for list processing to replace anonymous lambdas
    def parse_list_items(item_list):
        return [parse_single_ingredient(item) for item in item_list]

    # Process ingredients and name the column 'final_ingredients' directly
    scraped_df['final_ingredients'] = scraped_df['ingredients'].apply(parse_list_items)
    
    # Rename 'directions' to 'structured_directions' to match schema
    # (Scraped directions are already lists, so just renaming is sufficient)
    scraped_df.rename(columns={'directions': 'structured_directions'}, inplace=True)
    
    # --- FINAL CLEANUP ---
    # Infer source if missing (usually present, but safe to default)
    if 'source' not in scraped_df.columns:
        scraped_df['source'] = filename.split('_')[1]
        
    # Select only the standard columns, dropping raw 'ingredients'
    final_df = scraped_df[config.OUTPUT_COLUMNS]
    
    final_df.to_csv(output_path, index=False)
    return final_df

# --- Execution Entry Point ---

def main():
    # 1. Execute RecipeNLG processing
    nlg_results = process_recipenlg_dataset()
    if nlg_results is not None:
        print("\n--- SAMPLE OUTPUT: RecipeNLG (Processed) ---")
        # Display key columns to verify parsing logic
        print(nlg_results[['title', 'final_ingredients']].head(3).to_string(index=False))

    # 2. Execute processing for all scraped JSON data
    # Iterate over the file list defined in config
    for current_file in config.SCRAPED_FILES_TO_PROCESS:
        scraped_results = process_scraped_json_file(current_file)
        if scraped_results is not None:
            print(f"\n--- SAMPLE OUTPUT: {current_file} ---")
            # Display title and the new structured ingredient format
            print(scraped_results[['title', 'final_ingredients']].head(2).to_string(index=False))

    print("\nProcessing workflow completed successfully.")

if __name__ == "__main__":
    main()