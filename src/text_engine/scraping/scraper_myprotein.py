import requests
from bs4 import BeautifulSoup
import json
import re 
import time
import os
import sys
from src.text_engine import config

# --- Project Path Definitions ---
OUTPUT_FILE = os.path.join(config.RAW_DATA_DIR, "myprotein_recipes_ALL.json")

# --- Constants ---

BASE_URL = "https://www.myprotein.com"
RECIPE_LIST_URL = f"{BASE_URL}/thezone/recipe/"
HEADERS = {'User-Agent': 'TFG-Recipe-Scraper/1.0'}

# --- Scraping Functions ---

def get_recipe_links(page_url: str) -> list[str]:
    """
    Retrieves all recipe URLs from a pagination page.
    """
    try:
        response = requests.get(page_url, headers=HEADERS)
        # raise_for_status() will trigger an exception if code is 404 or 500
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        recipe_cards = soup.select("div.post-preview")
        
        if not recipe_cards:
            return []
            
        print(f"  Found {len(recipe_cards)} recipe cards.")

        for card in recipe_cards:
            heading = card.select_one("h2")
            if heading:
                # The <a> tag is the parent of the <h2> title in this specific site structure
                link_tag = heading.find_parent("a")
                if link_tag and link_tag.has_attr('href'):
                    href = link_tag['href']
                    # Construct full URL if the link is relative
                    if not href.startswith('http'):
                        href = f"{BASE_URL}{href}"
                    links.append(href)
            
        return list(set(links))
        
    except requests.exceptions.RequestException as e:
        # A 404 error is our signal that we've reached the last page
        if e.response and e.response.status_code == 404:
            print(f"  Info: Page {page_url} not found (404). End of pagination.")
        else:
            print(f"  HTTP Error: {e}")
        return []

def scrape_recipe_page(recipe_url: str) -> dict | None:
    """
    Extracts title, ingredients, and directions from a specific recipe page.
    """
    try:
        response = requests.get(recipe_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Title
        title_tag = soup.select_one("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Title not found"
        
        # 2. Ingredients
        ingredients = []
        # Search for an <h2> containing the word 'Ingredients' (case-insensitive). Case insensitive means we don't care if it's uppercase or lowercase.
        ingredients_h2 = soup.find("h2", string=re.compile(r'Ingredients', re.IGNORECASE))
        
        if ingredients_h2:
            # The list is usually the next sibling <ul> after the header <h2>
            ingredients_list_ul = ingredients_h2.find_next_sibling("ul")
            if ingredients_list_ul:
                items = ingredients_list_ul.find_all("li")
                for item in items:
                    ingredients.append(item.get_text(strip=True))

        # 3. Directions
        directions = []
        instructions_h2 = soup.find("h2", string=re.compile(r'(Instructions|Method)', re.IGNORECASE))
        
        if instructions_h2:
            # find_next_sibling is a BeautifulSoup method that finds the next sibling tag. A sibling is a tag at the same level in the HTML hierarchy.
            steps_list_ol = instructions_h2.find_next_sibling("ol")
            if steps_list_ol:
                steps = steps_list_ol.find_all("li")
                for step in steps:
                    directions.append(step.get_text(strip=True))

        return {
            "title": title,
            "ingredients": ingredients,
            "directions": directions,
            "link": recipe_url,
            "source": "Myprotein"
        }
        
    except Exception as e:
        print(f"  Parsing error in {recipe_url}: {e}")
        return None

# --- Main Execution ---

def main():
    print("Starting MyProtein scraper (with pagination)...")
    
    all_recipe_links = []
    page_number = 1
    
    # Pagination Loop
    while True:
        if page_number == 1:
            page_url = RECIPE_LIST_URL
        else:
            page_url = f"{RECIPE_LIST_URL}page/{page_number}/"
            
        print(f"\n--- Scanning Page {page_number} ---")
        
        links_on_this_page = get_recipe_links(page_url)
        
        if not links_on_this_page:
            print("No more links found. Ending pagination.")
            break 
        
        all_recipe_links.extend(links_on_this_page)
        page_number += 1
        time.sleep(0.5) 

    total_unique_links = list(set(all_recipe_links))
    print(f"\nFound {len(total_unique_links)} unique recipes.")
    print("Extracting details...")
    
    all_recipes = []
    for i, link in enumerate(total_unique_links):
        print(f"Processing {i+1}/{len(total_unique_links)}: {link}")
        data = scrape_recipe_page(link)
        if data:
            all_recipes.append(data)
        time.sleep(0.2)
            
    print(f"\nSaving {len(all_recipes)} recipes to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_recipes, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()