import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os
import sys
from src.text_engine import config

# --- File Paths ---
OUTPUT_FILE = os.path.join(config.RAW_DATA_DIR, "scraped_airfryer.json")

# --- Constants (Specific to this scraper) ---
BASE_URL = "https://recetasparafreidoradeaire.com"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'es-ES,es;q=0.9',
}

# --- Helper Function ---
def clean_text(text: str) -> str:
    """
    1. Truncates text immediately if an HTML tag opener '<' is found.
    2. Removes leading emojis, numbers, and symbols.
    """
    if not text:
        return ""
    
    # 1. Aggressive Cut: If we see a '<' character, assume it is the start of an HTML tag artifact
    # and cut everything after it. This solves the <div style...> issue.
    if '<' in text:
        text = text.split('<')[0]
    
    # 2. Clean Start: Remove emojis, numbers (1., 1) and symbols from the beginning.
    # ^ = Start of string
    # [\d\W_]+ = Matches digits, non-word chars (emojis), and underscores
    text = re.sub(r'^[\d\W_]+', '', text)
    
    return text.strip()

# --- Scraping Functions ---

def get_recipe_links(page_url: str) -> list[str]:
    """
    Scans a listing page to extract recipe URLs based on the 'card-title' structure.
    """
    try:
        print(f"  Scanning: {page_url}")
        response = requests.get(page_url, headers=HEADERS, timeout=10)
        
        if response.status_code == 404:
            print("  Info: Page not found (404). End of list.")
            return []
            
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        
        # <h4 class="card-title"> <a href="...">
        card_titles = soup.select("h4.card-title a")
        
        for link_tag in card_titles:
            href = link_tag.get('href')
            if href:
                if not href.startswith('http'):
                    href = f"{BASE_URL}{href}" if href.startswith('/') else f"{BASE_URL}/{href}"
                
                # Filter to keep only blog posts
                if "/blog/" in href:
                    links.append(href)
            
        print(f"  Found {len(links)} links on this page.")
        return list(set(links))
        
    except Exception as e:
        print(f"  Error scanning {page_url}: {e}")
        return []

def scrape_airfryer_recipe(recipe_url: str) -> dict | None:
    """
    Extracts data based on the specific DOM structure provided in the screenshots.
    """
    try:
        response = requests.get(recipe_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Title Extraction
        # Targeted selector: h1 with class "card-title"
        title_tag = soup.find("h1", class_="card-title")
        title = title_tag.get_text(strip=True) if title_tag else "Title not found"
        
        # 2. Ingredients Extraction
        ingredients = []
        # Logic: Find h2 containing "Ingredientes" and get the UL that follows
        ing_header = soup.find(lambda tag: tag.name == "h2" and "Ingredientes" in tag.get_text())
        
        if ing_header:
            # The list is in the next sibling container or directly after
            # We look for the next <ul> tag
            ing_ul = ing_header.find_next("ul")
            if ing_ul:
                # Iterate over <li> items. Clean emojis and extract text from <p> or <strong>
                for li in ing_ul.find_all("li"):
                    text = li.get_text(strip=True)
                    # Clean up common emojis often used in these lists (optional but recommended)
                    text = re.sub(r'[^\w\s\d,./ñáéíóúÁÉÍÓÚ-]', '', text).strip()
                    if len(text) > 1: # Avoid empty bullets
                        ingredients.append(text)

        # 3. Directions Extraction
        # Logic: Steps are fragmented into <h3> (Title) and <p> (Description) pairs
        directions = []
        
        # Find the start of the section: <h2> "Preparación paso a paso"
        steps_header = soup.find(lambda tag: tag.name == "h2" and "Preparación" in tag.get_text())
        
        if steps_header:
            # We iterate through siblings after the header
            current_element = steps_header.find_next_sibling()
            
            # Stop if we hit footer, another h2, or run out of elements
            while current_element and current_element.name != 'h2':
                
                # steps titles are in <h3> (e.g., "1 Activar la levadura")
                if current_element.name == 'h3':
                    raw_title = current_element.get_text(strip=True)
                    # Clean title (removes "1." or emojis)
                    step_title = clean_text(raw_title)
                    
                    # The description is usually in the immediate next <p>
                    description_p = current_element.find_next_sibling('p')
                    if description_p:
                        # Get text but be careful of inner tags
                        raw_desc = description_p.get_text(" ", strip=True)
                        # Clean description (cuts at HTML tags like <div...)
                        step_desc = clean_text(raw_desc)
                        
                        # Combine "Title: Description"
                        full_step = f"{step_title}: {step_desc}"
                        directions.append(full_step)
                
                current_element = current_element.find_next_sibling()

        return {
            "title": title,
            "ingredients": ingredients,
            "directions": directions,
            "link": recipe_url,
            "source": "AirFryerRecipes"
        }
        
    except Exception as e:
        print(f"  Error parsing {recipe_url}: {e}")
        return None

# --- Main Execution ---

def main():
    print("Starting AirFryer website scraper (Fixed Structure)...")
    
    all_links = []
    page = 1
    max_pages = 50 # Increase if needed
    
    # PHASE 1: Collect links
    while page <= max_pages:
        # Pagination format: /pagina/2
        url = BASE_URL if page == 1 else f"{BASE_URL}/pagina/{page}"
            
        print(f"\n--- Processing Page {page} ---")
        links = get_recipe_links(url)
        
        if not links:
            print("No more recipes found. Stopping pagination.")
            break
            
        all_links.extend(links)
        page += 1
        time.sleep(1)

    unique_links = list(set(all_links))
    print(f"\nCollected {len(unique_links)} unique recipe URLs.")
    
    # --- PHASE 2: Extract details ---
    results = []
    print("Extracting details from recipes...")
    
    for i, link in enumerate(unique_links):
        print(f"[{i+1}/{len(unique_links)}] Parsing: {link}")
        data = scrape_airfryer_recipe(link)
        
        # Strict Filtering Logic
        # 1. Must have data
        # 2. Must have ingredients list
        # 3. Must have more than one step (Discard single-element directions)
        # 4. Must not contain site-name artifacts in ingredients
        if data and data['ingredients'] and len(data['directions']) > 1:
            if "Recetasparafreidora" not in data['ingredients']:
                results.append(data)
                print(f"  Success: {data['title']} saved.")
            else:
                print(f"  Skipped: Scraping artifact found in ingredients.")
        else:
            print(f"  Skipped: Insufficient data or single-step directions.")
            
    # PHASE 3: Save
    print(f"\nSaving {len(results)} recipes to {OUTPUT_FILE}...")
    
    # Directory creation assumes parent folder exists or is handled by config setup
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Scraping workflow completed successfully.")

if __name__ == "__main__":
    main()