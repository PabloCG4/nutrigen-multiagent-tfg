import pandas as pd
import ast
import sys
import csv
import os
from src.text_engine import config

# Standard schema for the unified Master DataFrame
FINAL_COLUMNS = config.OUTPUT_COLUMNS

# --- 2. Helper Functions ---

def safe_convert_to_list(column_data):
    """
    Safely converts string representations of lists back into Python objects.
    Also harmonizes ingredient dictionary keys from 'name' to 'ingredient'.
    """
    try:
        # Convert to list using ast.literal_eval
        data = ast.literal_eval(str(column_data))
        if not isinstance(data, list):
            return []
        
        # Ensure all ingredient dicts use the key 'ingredient'
        # (Scraped files use 'name', RecipeNLG uses 'ingredient')
        normalized_data = []
        for item in data:
            if isinstance(item, dict):
                # Get the name regardless of which key it uses
                name = item.get('name') or item.get('ingredient') or "Unknown"
                
                # Reconstruct dictionary with standardized keys
                normalized_data.append({
                    "ingredient": name,
                    "quantity": item.get('quantity', 1.0),
                    "unit": item.get('unit', 'unit')
                })
            else:
                normalized_data.append(item)
        return normalized_data
    except (ValueError, SyntaxError, TypeError):
        return []

# --- 3. Data Loading and Standardization ---

def load_processed_dataframes():
    """
    Loads and standardizes the processed sources into a common format.
    """
    standardized_dfs = []

    # --- Source 1: RecipeNLG (Kaggle) ---
    print(f"Loading {os.path.basename(config.PROCESSED_RECIPENLG_PATH)}...")
    try:
        df_nlg = pd.read_csv(config.PROCESSED_RECIPENLG_PATH)
        # Apply safe conversion to list objects
        df_nlg['final_ingredients'] = df_nlg['final_ingredients'].apply(safe_convert_to_list)
        df_nlg['structured_directions'] = df_nlg['structured_directions'].apply(safe_convert_to_list)
        
        standardized_dfs.append(df_nlg[FINAL_COLUMNS])
        print(f"Loaded RecipeNLG: {len(df_nlg)} rows.")
    except Exception as e:
        print(f"Warning: Could not process RecipeNLG. Error: {e}")

    # --- Source 2: Scraped Data ---
    # UPDATED: Added AirFryer and Ninja to the loading list using config paths
    scraped_files = [
        (config.PROCESSED_PALADAR_PATH, "DirectoAlPaladar"),
        (config.PROCESSED_MYPROTEIN_PATH, "MyProtein"),
        (config.PROCESSED_AIRFRYER_PATH, "AirFryerRecipes"),
        (config.PROCESSED_NINJA_PATH, "NinjaTestKitchen")
    ]

    for file_path, source_name in scraped_files:
        print(f"Loading {os.path.basename(file_path)}...")
        try:
            # Check if file exists before trying to read (avoid hard crash)
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}. Skipping.")
                continue

            df = pd.read_csv(file_path)
            
            # Columns are now ALREADY standardized by process_recipes.py
            # No need to map 'ingredients_processed' -> 'final_ingredients' anymore.
            # We just verify and convert strings to lists.
            
            if 'final_ingredients' in df.columns:
                df['final_ingredients'] = df['final_ingredients'].apply(safe_convert_to_list)
            
            if 'structured_directions' in df.columns:
                df['structured_directions'] = df['structured_directions'].apply(safe_convert_to_list)
            
            # Select strictly the final columns
            standardized_dfs.append(df[FINAL_COLUMNS])
            print(f"Loaded {source_name}: {len(df)} rows.")
        except Exception as e:
            print(f"Warning: Could not process {source_name}. Error: {e}")

    return standardized_dfs

# --- 4. Main Execution ---

def main():
    """
    Orchestrates unification, deduplication, and saving of the Master dataset.
    """
    dfs = load_processed_dataframes()
    
    if not dfs:
        print("Error: No dataframes were loaded. Check your processed files.")
        sys.exit(1)

    # --- Unification ---
    print("\nCombining datasets...")
    master_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows before deduplication: {len(master_df)}")

    # --- Deduplication ---
    # We use 'link' as the unique identifier to remove duplicates
    # Redundancy was intentionally preserved to avoid losing the varied instructions and nutritional approaches across different datasets
    print("Removing duplicates based on 'link'...")
    master_df.drop_duplicates(subset=['link'], keep='first', inplace=True)
    print(f"Total unique recipes: {len(master_df)}")

    # --- Save Master Dataset ---
    print(f"\nSaving Master Dataset to {config.MASTER_RECIPE_PATH}...")
    try:
        master_df.to_csv(
            config.MASTER_RECIPE_PATH, 
            index=False, 
            encoding='utf-8', 
            quoting=csv.QUOTE_ALL  # To handle commas in text fields properly
        )
        print(f"Success! Master dataset saved with {len(master_df)} recipes.")
    except Exception as e:
        print(f"Error saving master file: {e}")

    # --- Summary ---
    print("\n--- Master Dataset Summary ---")
    print(master_df.info())
    print("\nSample (first 3 rows):")
    print(master_df.head(3))

if __name__ == "__main__":
    main()