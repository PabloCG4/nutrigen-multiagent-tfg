import requests
import json
import logging
import time
from src.vision import config

# --- CONFIGURATION ---
TAXONOMY_URL = "https://static.openfoodfacts.org/data/taxonomies/categories.json"

# Categories from the website that we want to map to the official taxonomy.
TARGET_CATEGORIES_ES = [
    "Alimentos y bebidas de origen vegetal", "Alimentos de origen vegetal", "Botanas", "Snacks dulces", "Lácteos",
    "Cereales y patatas", "Bebidas", "Productos a base de carne", "Comidas fermentadas", 
    "Frutas y verduras y sus productos", "Productos fermentados de la leche", "Carnes", "Embutidos",
    "Cereales y derivados", "Condimentos", "Quesos", "Conservas", "Galletas y pasteles", "Desayunos",
    "Postres", "Untables", "Verduras y hortalizas y sus productos", "Productos del mar", "Dulces",
    "Comidas preparadas", "Bebidas de origen vegetal", "Pescados", "Bebidas y preparaciones de bebidas",
    "Productos del olivo", "Aceites y grasas", "Salsas", "Untables dulces", "Cacao y sus productos",
    "Grasas vegetales", "Jamón", "Panes", "Hortalizas", "Galletas", "Aceites vegetales",
    "Alimentos de origen vegetal en conserva", "Productos agrícolas", "Frutas y sus productos",
    "Postres lácteos", "Snacks salados", "Congelados", "Aceites de oliva", "Untables vegetales",
    "Leguminosas y derivados", "Endulzantes", "Frutos de cáscara y derivados", "Semillas",
    "Aceites de oliva virgen extra", "Aperitivos", "Chocolates", "Yogures", "Aceites de oliva virgen",
    "Conservas de pescado", "Bebidas a base de frutas", "Alimentos festivos", "Suplementos dietéticos",
    "Bebidas alcohólicas", "Verduras y hortalizas en conserva", "Leguminosas", "Gastronomía navideña",
    "Productos apícolas", "Cereales para el desayuno", "Frutos de cáscara", "Pasteles", "Mieles",
    "Dulces de Navidad", "Productos deshidratados", "Zumos y néctares", "Pastas alimenticias",
    "Jamón de país", "Aves", "Jamones secos", "Productos cárnicos españoles", "Chips fritos",
    "Encurtidos", "Cafés", "Jamón cocido", "Jamón Serrano", "Postres helados",
    "Preparaciones a base de frutas y verduras", "Turrones", "Vegetales encurtidos", "Frutas",
    "Bebidas para tomar calientes", "Verduras de tallo", "Confitura", "Atunes", "Quesos de vaca",
    "Helados y sorbetes"
]

def main():
    # DOWNLOAD TAXONOMY
    try:
        response = requests.get(TAXONOMY_URL)
        response.raise_for_status()
        taxonomy_data = response.json()
        print(" Taxonomy downloaded successfully.")
    except Exception as e:
        print(f" Failed to download taxonomy: {e}")
        return

    # BUILD LOOKUP TABLE
    # Mapping structure: "name_lowercase" -> "en:technical-id"
    name_to_id_map = {}
    
    for tag_id, details in taxonomy_data.items():
        names = details.get('name', {})
        
        # Index spanish and english names
        name_es = names.get('es')
        if name_es:
            name_to_id_map[name_es.lower()] = tag_id
        
        name_en = names.get('en')
        if name_en:
            name_to_id_map[name_en.lower()] = tag_id

    print(f" Lookup table built with {len(name_to_id_map)} entries.")

    # RESOLVE TARGET CATEGORIES
    resolved_ids = []
    failed_items = []

    for item in TARGET_CATEGORIES_ES:
        item_lower = item.lower()
        
        # Exact match in lookup table
        found_id = name_to_id_map.get(item_lower)
        
        if found_id:
            print(f"'{item}' -> '{found_id}'")
            resolved_ids.append(found_id)
        else:
            print(f" No exact translation found for '{item}'")
            # Fallback: Construct a potential ID using 'en:' prefix
            fallback_id = "en:" + item.lower().replace(" ", "-")
            resolved_ids.append(fallback_id)
            failed_items.append(item)

    # --- OUTPUT GENERATION ---

    print("CATEGORIES = [")
    for valid_id in resolved_ids:
        print(f' "{valid_id}",')
    print("]")

    if failed_items:
        print(f"{len(failed_items)} categories were not found in the dictionary.")

if __name__ == "__main__":
    main()