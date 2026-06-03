import os
import requests
import json
import time
import logging
import unicodedata
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter 
from src.vision import config

# --- CONFIGURATION ---
BASE_API_URL = "https://es.openfoodfacts.org/category"
IMAGES_ROOT_DIR = Path(config.IMAGES_DIR)

# Number of simultaneous threads
MAX_WORKERS = 25

REQUEST_HEADERS = {
    'User-Agent': 'TFG-PersonalChef-Project/1.0'
}

# --- GLOBAL SESSION SETUP ---
GLOBAL_SESSION = requests.Session()
GLOBAL_SESSION.headers.update(REQUEST_HEADERS)

# Increase the pool size to match our number of workers (25)
adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
GLOBAL_SESSION.mount("https://", adapter)
GLOBAL_SESSION.mount("http://", adapter)

# Global Index for Smart Cache. Maps barcodes to local file paths for instant copying if the same product appears in multiple categories.
GLOBAL_BARCODE_INDEX = {}

# --- CATEGORY MAPPING using taxonomy_mapper.py output ---
CATEGORY_MAPPING = {
    #"Alimentos y bebidas de origen vegetal": "en:plant-based-foods-and-beverages",
    "Alimentos de origen vegetal": "en:plant-based-foods",
    #"Botanas": "en:snacks",
    #"Snacks dulces": "en:sweet-snacks",
    #"Lácteos": "en:dairies",
    #"Cereales y patatas": "en:cereals-and-potatoes",
    #"Bebidas": "en:beverages",
    #"Productos a base de carne": "en:meats-and-their-products",
    #"Comidas fermentadas": "en:fermented-foods",
    #"Frutas y verduras y sus productos": "en:fruits-and-vegetables-based-foods",
    #"Productos fermentados de la leche": "en:fermented-milk-products",
    #"Carnes": "en:meats",
    #"Embutidos": "en:prepared-meats",
    #"Cereales y derivados": "en:cereals-and-their-products",
    #"Condimentos": "en:condiments",
    #"Quesos": "en:cheeses",
    #"Conservas": "en:canned-foods",
    #"Galletas y pasteles": "en:biscuits-and-cakes",
    #"Desayunos": "en:breakfasts",
    #"Postres": "en:desserts",
    #"Untables": "en:spreads",
    #"Verduras y hortalizas y sus productos": "en:vegetables-based-foods",
    #"Productos del mar": "en:seafood",
    #"Dulces": "en:confectioneries",
    #"Comidas preparadas": "en:meals",
    #"Bebidas de origen vegetal": "en:plant-based-beverages",
    #"Pescados": "en:fishes",
    #"Bebidas y preparaciones de bebidas": "en:beverages-and-beverages-preparations",
    #"Productos del olivo": "en:olive-tree-products",
    #"Aceites y grasas": "en:fats",
    #"Salsas": "en:sauces",
    #"Untables dulces": "en:sweet-spreads",
    #"Cacao y sus productos": "en:cocoa-and-its-products",
    #"Grasas vegetales": "en:vegetable-fats",
    #"Jamón": "en:hams",
    #"Panes": "en:breads",
    #"Hortalizas": "en:vegetables",
    #"Galletas": "en:biscuits",
    #"Aceites vegetales": "en:vegetable-oils",
    #"Alimentos de origen vegetal en conserva": "en:canned-plant-based-foods",
    #"Productos agrícolas": "en:farming-products",
    #"Frutas y sus productos": "en:fruits-based-foods",
    #"Postres lácteos": "en:dairy-desserts",
    #"Snacks salados": "en:salty-snacks",
    #"Frozen foods": "en:frozen-foods",
    #"Aceites de oliva": "en:olive-oils",
    #"Untables vegetales": "en:plant-based-spreads",
    #"Leguminosas y derivados": "en:legumes-and-their-products",
    #"Endulzantes": "en:sweeteners",
    #"Frutos de cáscara y derivados": "en:nuts-and-their-products",
    #"Semillas": "en:seeds",
    #"Aceites de oliva virgen extra": "en:extra-virgin-olive-oils",
    #"Aperitivos": "en:appetizers",
    #"Chocolates": "en:chocolates",
    #"Yogures": "en:yogurts",
    #"Aceites de oliva virgen": "en:virgin-olive-oils",
    #"Conservas de pescado": "en:canned-fishes",
    #"Bebidas a base de frutas": "en:fruit-based-beverages",
    #"Alimentos festivos": "en:festive-foods",
    #"Suplementos dietéticos": "en:dietary-supplements",
    #"Bebidas alcohólicas": "en:alcoholic-beverages",
    #"Verduras y hortalizas en conserva": "en:canned-vegetables",
    #"Leguminosas": "en:legumes",
    #"Gastronomía navideña": "en:christmas-foods-and-drinks",
    #"Productos apícolas": "en:bee-products",
    #"Cereales para el desayuno": "en:breakfast-cereals",
    #"Frutos de cáscara": "en:nuts",
    #"Pasteles": "en:cakes",
    #"Mieles": "en:honeys",
    #"Dulces de Navidad": "en:christmas-sweets",
    #"Productos deshidratados": "en:dried-products",
    #"Zumos y néctares": "en:juices-and-nectars",
    #"Pastas alimenticias": "en:pastas",
    #"Jamón de país": "en:cured-hams",
    #"Aves": "en:poultries",
    #"Jamones secos": "en:dried-hams",
    #"Productos cárnicos españoles": "en:spanish-meat-products",
    #"Chips fritos": "en:crisps",
    #"Encurtidos": "en:pickles",
    #"Cafés": "en:coffees",
    #"Jamón cocido": "en:white-hams",
    #"Jamón Serrano": "en:serrano-ham",
    #"Postres helados": "en:frozen-desserts",
    #"Preparaciones a base de frutas y verduras": "en:fruit-and-vegetable-preserves",
    #"Turrones": "en:turron",
    #"Vegetales encurtidos": "en:plant-based-pickles",
    #"Frutas": "en:fruits",
    #"Bebidas para tomar calientes": "en:hot-beverages",
    #"Verduras de tallo": "en:vegetable-rods",
    #"Confitura": "en:jams",
    #"Atunes": "en:tunas",
    #"Quesos de vaca": "en:cow-cheeses",
    #"Helados y sorbetes": "en:ice-creams-and-sorbets"
}

# --- CORE FUNCTIONS ---

def sanitize_folder_name(name):
    # Remove accents and special characters for safe folder names
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return clean_name.strip().replace(" ", "_")


def build_local_index():
    '''
    Scans the existing images directory and builds a global index of barcodes to file paths.
    This allows the program to quickly check if an image already exists locally before attempting to download it again, enabling instant local copies for products that appear in multiple categories.
    '''

    count = 0
    # rglob to find all image files in the directory and subdirectories
    for path in IMAGES_ROOT_DIR.rglob("*"):
        if path.is_file() and path.suffix in ['.jpg', '.png']:
            barcode = path.stem 
            # global_barcode_index is updated with existing files to avoid redundant downloads
            if barcode not in GLOBAL_BARCODE_INDEX:
                GLOBAL_BARCODE_INDEX[barcode] = path
                count += 1
    print(f"Index ready: {count} images found locally.")

def download_or_copy_task(task_data):
    """
    Worker function using GLOBAL_SESSION with proper ADAPTER settings.
    task_data is a dictionary containing 'url', 'path', 'name', and optionally 'source_local_path' for instant copying.
    """
    image_url = task_data['url']
    destination_path = task_data['path']
    product_name = task_data['name']
    # Check if the file already exists
    source_local_path = task_data.get('source_local_path')

    if destination_path.exists():
        return True

    # LOCAL COPY (Instant)
    if source_local_path and source_local_path.exists():
        try:
            # shutil.copy2 preserves metadata and is faster than downloading if the file already exists locally
            shutil.copy2(source_local_path, destination_path)
            print(f"Copy local: {product_name}")
            return True
        except Exception:
            pass 

    # NETWORK DOWNLOAD
    max_retries = 10
    current_timeout = 60
    
    # Only if the file doesn't exist locally
    for attempt in range(1, max_retries + 1):
        try:
            # Using GLOBAL_SESSION
            response = GLOBAL_SESSION.get(image_url, timeout=current_timeout)
            
            if response.status_code == 200:
                # Open in binary mode and write the content
                image_file = open(destination_path, 'wb')
                # Write the content downloaded from the internet
                image_file.write(response.content)
                image_file.close()
                
                print(f" Saved: {product_name}")
                return True
            
            elif response.status_code in [404, 403]:
                return False

        except Exception:
             pass

        if attempt < max_retries:
            time.sleep(1)

    print(f" FAILED: {product_name}")
    return False

def get_best_image_url(product_data):
    ""
    "OpenFoodFacts products can have multiple image fields. This function tries them in order of preference."
    ""
    url = product_data.get('image_front_url')
    if not url:
        url = product_data.get('image_url')
    if not url:
        selected = product_data.get('selected_images', {})
        url = selected.get('front', {}).get('display', {}).get('es') 
    return url

def process_single_category(category_name_es, category_id_api):
    """
    Orchestrates the download process for a specific category.
    It handles pagination, retries on network failures, and manages parallel downloads.
    """
    # SETUP
    folder_name = sanitize_folder_name(category_name_es)
    category_folder = IMAGES_ROOT_DIR / folder_name
    category_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing Category: {category_name_es}")

    current_page = 1
    
    # PAGINATION LOOP
    while True:
        # We request 50 products at a time to minimize API calls
        api_endpoint = f"{BASE_API_URL}/{category_id_api}.json?page={current_page}&page_size=50"
        
        page_fetch_successful = False
        products_list = []
        
        # FETCH DATA
        for attempt in range(1, 4):
            try:
                # Use the high-speed Global Session
                api_response = GLOBAL_SESSION.get(api_endpoint, timeout=60)
                
                # We reached the end of the category
                if api_response.status_code == 404:
                    print("No more pages to fetch.")
                    return # Exit function, job done for this category

                # api_response is a requests.Response object, we need to call .json() to get the actual data. The data is expected to be a dictionary with a 'products' key containing a list of products.
                json_data = api_response.json()
                # the list of products is extracted from the 'products' key in the JSON response. Each product is expected to be a dictionary containing details about the product, including its barcode, name, and image URLs.
                products_list = json_data.get('products', [])
                page_fetch_successful = True
                break 
                
            except Exception as e:
                time.sleep(1) 
        
        if not page_fetch_successful:
            print(f" Unable to fetch Page {current_page}. Moving to next category.")
            break 
            
        # The page exists but has no products inside
        if not products_list:
            print(f" No products found.")
            break
            
        # PREPARE TASKS 
        tasks = []
        
        for product in products_list:
            barcode = product.get('code')
            name = product.get('product_name', 'Unknown Product')
            if not name: name = "Unknown Product"
            
            img_url = get_best_image_url(product)
            
            if not img_url or not barcode:
                continue 

            # File Extension Logic (Handle weird URLs like image.php?id=123)
            raw_extension = img_url.split('.')[-1]
            if '?' in raw_extension or len(raw_extension) > 4:
                clean_extension = "jpg"
            else:
                clean_extension = raw_extension
                
            file_name = f"{barcode}.{clean_extension}"
            full_save_path = category_folder / file_name

            if full_save_path.exists():
                # Verify it's in our global index so future categories can copy from here if needed.
                if barcode not in GLOBAL_BARCODE_INDEX:
                    GLOBAL_BARCODE_INDEX[barcode] = full_save_path
                continue
            
            # Check if we have this barcode in the global index (from previous categories)
            source_local = GLOBAL_BARCODE_INDEX.get(barcode)

            # Create the order for the worker thread
            task = {
                'url': img_url,
                'path': full_save_path,
                'name': name,
                'source_local_path': source_local 
            }
            tasks.append(task)

        # EXECUTE TASKS IN PARALLEL
        if tasks:
            print(f"\n Page {current_page}: Processing {len(tasks)} items")
            
            # Launch parallel workers
            executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
            # We use list() to force the program to wait until all images are downloaded
            list(executor.map(download_or_copy_task, tasks))
            executor.shutdown(wait=True)
            
            # Now that we have new files, add them to the index
            for task in tasks:
                if task['path'].exists():
                    # stem (root) is the filename without extension, which is the barcode
                    barcode = task['path'].stem
                    GLOBAL_BARCODE_INDEX[barcode] = task['path']
        else:
            # If tasks list is empty, it means we had everything locally already
            print(f"\n Page {current_page}: All products cached locally.")

        current_page += 1
        time.sleep(1)

if __name__ == "__main__":
    build_local_index()

    for category_name, category_id in CATEGORY_MAPPING.items():
        process_single_category(category_name, category_id)