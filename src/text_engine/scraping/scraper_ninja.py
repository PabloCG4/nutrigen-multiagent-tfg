import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os
import sys
from src.text_engine import config

# --- Project Path Definitions ---
OUTPUT_FILE = os.path.join(config.RAW_DATA_DIR, "scraped_ninja.json")

# --- Constants ---
BASE_URL = "https://ninjatestkitchen.eu"
SEARCH_BASE_URL = "https://ninjatestkitchen.eu/search/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-GB,en;q=0.9',
}

# --- Helper Function ---
def clean_text(text: str) -> str:
    """
    Cleans text by stripping excessive whitespace.
    """
    if not text:
        return ""
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Scraping Functions ---

def get_recipe_links(page_number: int) -> list[str]:
    """
    Extracts recipe URLs from a specific pagination page.
    """
    if page_number == 1:
        url = SEARCH_BASE_URL
    else:
        url = f"{SEARCH_BASE_URL}?_paged={page_number}"

    try:
        print(f"  Scanning Page {page_number}: {url}")
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 404:
            print("  Info: Page not found (404).")
            return []
            
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        card_links = soup.select("a.recipe-card__link")
        
        for link in card_links:
            href = link.get('href')
            if href:
                if not href.startswith('http'):
                    href = f"{BASE_URL}{href}" if href.startswith('/') else f"{BASE_URL}/{href}"
                links.append(href)
            
        print(f"  Found {len(links)} links on this page.")
        return list(set(links))
        
    except Exception as e:
        print(f"  Error scanning page {page_number}: {e}")
        return []

def scrape_ninja_recipe(recipe_url: str) -> dict | None:
    """
    Parses recipe details with robust handling for nested tags like <strong> and <br>.
    """
    try:
        response = requests.get(recipe_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Title Extraction
        title_tag = soup.find("h1")
        if title_tag:
            title = clean_text(title_tag.get_text())
        else:
            meta_title = soup.find("meta", attrs={"itemprop": "name"})
            title = meta_title["content"] if meta_title else "Title not found"
        
        # 2. Ingredients Extraction
        ingredients = []
        # We target the list directly. BeautifulSoup finds it regardless of the parent div.
        ing_list = soup.select_one("ul.single-ingredients__list")
        
        if ing_list:
            items = ing_list.find_all("li")
            for item in items:
                # separator=' ' ensures words don't stick together if <span> or <br> are used
                text = clean_text(item.get_text(separator=' ', strip=True))
                if text:
                    ingredients.append(text)

        # 3. Directions Extraction (Improved for image_7a083f.png)
        directions = []
        method_list = soup.select_one("ul.single-method__method")
        
        if method_list:
            items = method_list.find_all("li")
            for item in items:
                step_header = item.find("h6")
                
                # Retrieve ALL paragraphs in the list item, not just the first one.
                # This fixes cases where instructions are split or wrapped in <div>/<strong>
                p_tags = item.find_all("p")
                
                if p_tags:
                    step_title = clean_text(step_header.get_text()) if step_header else "Step"
                    
                    # Join all paragraphs, using separator=' ' to handle <br> and <strong> correctly
                    instruction_parts = [clean_text(p.get_text(separator=' ', strip=True)) for p in p_tags]
                    instruction = " ".join(filter(None, instruction_parts))
                    
                    if instruction:
                        full_step = f"{step_title}: {instruction}"
                        directions.append(full_step)

        return {
            "title": title,
            "ingredients": ingredients,
            "directions": directions,
            "link": recipe_url,
            "source": "NinjaTestKitchen"
        }
        
    except Exception as e:
        print(f"  Error parsing {recipe_url}: {e}")
        return None

# --- Main Execution ---

def main():
    print("Starting Ninja Test Kitchen scraper...")
    
    all_links = []
    page = 1
    max_pages = 50 
    
    # PHASE 1: Pagination Loop
    while page <= max_pages:
        links = get_recipe_links(page)
        
        if not links:
            print("No more recipes found. Stopping pagination.")
            break
            
        all_links.extend(links)
        page += 1
        time.sleep(1) 

    unique_links = list(set(all_links))
    print(f"\nTotal collected: {len(unique_links)} unique recipes.")
    
    results = []
    print("Extracting details...")
    
    # PHASE 2: Extraction Loop
    for i, link in enumerate(unique_links):
        print(f"[{i+1}/{len(unique_links)}] Parsing: {link}")
        data = scrape_ninja_recipe(link)
        
        # Validation with specific debug prints
        if data:
            has_ing = len(data['ingredients']) > 0
            has_dir = len(data['directions']) > 0
            
            if has_ing and has_dir:
                results.append(data)
                print(f"  Success: {data['title']}")
            else:
                # Debugging info
                missing = []
                if not has_ing: missing.append("ingredients")
                if not has_dir: missing.append("directions")
                print(f"  Skipped: Missing {', '.join(missing)}.")
        else:
            print("  Skipped: Error parsing.")
            
        time.sleep(0.5) 
            
    # PHASE 3: Save
    print(f"\nSaving {len(results)} recipes to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Scraping completed successfully.")

if __name__ == "__main__":
    main()