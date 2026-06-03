import requests
from bs4 import BeautifulSoup
import json
import time
from collections import deque # 'deque' is optimized for adding/removing items from both ends (perfect for queues)
import os
import sys
from src.text_engine import config

# --- Project Path Definitions ---
OUTPUT_FILE = os.path.join(config.RAW_DATA_DIR, "scraped_directoalpaladar_COMPLETO_v2.json")

# --- Scraping Constants ---

BASE_URL = "https://www.directoalpaladar.com"
HEADERS = {'User-Agent': 'TFG-Recipe-Scraper/1.0'}

# --- Link Extraction Functions ---

def get_all_category_urls(archive_url: str) -> list[str]:
    """
    Visits the archive/sitemap page to discover all recipe category URLs.
    """
    print(f"Searching for categories at: {archive_url}")
    try:
        response = requests.get(archive_url, headers=HEADERS)

        # Always check for HTTP 200 (OK). If the server is down or blocks us, we stop here.
        if response.status_code != 200:
            print(f"  Error downloading archive page. Code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Identify content boxes: The archive page has multiple sections (dates, tags, categories).
        # We need to find the specific <div> that holds the categories.
        all_category_boxes = soup.select('div.category')
        
        recipe_box = None
        # 2. Filter logic: Iterate through boxes to find the one explicitly titled "Recetas"
        for box in all_category_boxes:
            header = box.find('h3')
            if header and "recetas" in header.get_text(strip=True).lower():
                print(f"  Category container found: '{header.get_text(strip=True)}'.")
                recipe_box = box
                break

        if not recipe_box:
            print("  Warning: 'Recetas' container not found in archive.")
            return []

        # 3. Extract links: Get all <a> tags inside the identified box
        link_elements = recipe_box.select('ul li a')
        category_urls = []
        
        for link in link_elements:
            url = link.get('href')
            if url:
                # 4. Strict Filtering: We only want recipe categories, not generic tags.
                # We filter urls containing specific patterns like '/categoria/'
                if (url.startswith('/categoria/') or "/recetas-de-" in url) and url.startswith('/'):
                    if not url.startswith('http'):
                        url = BASE_URL + url
                    category_urls.append(url)

        # Use set() to remove potential duplicates immediately   
        unique_urls = list(set(category_urls))
        print(f"  Found {len(unique_urls)} filtered category URLs.")
        return unique_urls

    except Exception as e:
        print(f"  Unexpected error searching categories: {e}")
        return []

def get_recipe_urls_from_category(category_url: str) -> list[str]:
    """
    Visits a specific category page and extracts recipe URLs listed there.
    """
    print(f"Scanning category: {category_url}")
    try:
        response = requests.get(category_url, headers=HEADERS)
        if response.status_code != 200:
            print(f"  Error downloading category. Code: {response.status_code}")
            return []

        # BeautifulSoup organizes the HTML content for easy parsing
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # CSS Selector specific to Directo Al Paladar's article headlines
        recipe_link_selector = 'h2.abstract-title a'
        link_elements = soup.select(recipe_link_selector)

        recipe_urls = []
        for link in link_elements:
            url = link.get('href')
            if url:
                # Ensure absolute URL
                if not url.startswith('http'):
                    url = BASE_URL + url
                recipe_urls.append(url)

        unique_urls = list(set(recipe_urls))
        print(f"  Found {len(unique_urls)} recipe URLs.")
        return unique_urls

    except Exception as e:
        print(f"  Unexpected error scanning category: {e}")
        return []

# --- Main Parsing Function ---

def scrape_directoalpaladar_page(url: str) -> dict | list | None:
    """
    This function handles two types of pages:
    1. A Recipe Page: Returns a Dictionary with data.
    2. An Article/Listicle: Returns a List of new URLs to explore.
    3. Error/Invalid: Returns None.
    """
    print(f"Processing URL: {url}")
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"  Error: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Page Type Detection ---
        # We try to find elements that ONLY exist on a recipe page.
        title_element = soup.select_one('h1.post-title')
        ingredient_elements = soup.select('li.asset-recipe-list-item.m-is-ingr') 
        steps_element = soup.select_one('div.asset-recipe-steps')

        # Condition: It is a recipe if it has Title AND (Ingredients OR Steps)
        is_final_recipe = title_element is not None and (len(ingredient_elements) > 0 or steps_element is not None)

        if is_final_recipe:
            # --- Case A: Final Recipe ---
            # Extract structured data
            try:
                title = title_element.get_text(strip=True)
            except AttributeError:
                return None

            # Extract Ingredients: Loop through list items and clean text
            ingredients_list = []
            for item in ingredient_elements:
                name_el = item.select_one('.asset-recipe-ingr-name')
                amount_el = item.select_one('.asset-recipe-ingr-amount')
                
                # Strip is used to clean whitespace
                name = name_el.get_text(strip=True) if name_el else ""
                amount = amount_el.get_text(strip=True) if amount_el else ""
                
                # Combine amount and name (e.g., "500g" + "Flour")
                ingredients_list.append(f"{amount} {name}".strip())

            # Extract Directions: Get paragraphs <p> inside the steps container
            directions_list = []
            if steps_element: 
                directions_list = [step.get_text(strip=True) for step in steps_element.select('p')]

            # Time extraction
            try:
                total_time = soup.select_one('li.m-is-totaltime .asset-recipe-time-value').get_text(strip=True)
            except AttributeError:
                total_time = "Not specified"

            return {
                "title": title,
                "ingredients": ingredients_list,
                "directions": directions_list,
                "total_time": total_time,
                "link": url,
                "source": "DirectoAlPaladar"
            }

        else:
            # --- Case B: Index/Article (Find more links) ---
            # If it's not a recipe, we assume it might contain links to other recipes.
            # We look for outbound links inside the article text.
            link_elements = soup.select('a.text-outboundlink')
            found_urls = []
            
            # Whitelist patterns to ensure we only follow relevant links (no ads, no external sites)
            valid_patterns = ['/recetas-de-', '/receta-', '/postres/', '/directo-al-paladar/', '/categoria/']
            
            for link in link_elements:
                # Takes the href attribute in which the URL is stored
                href = link.get('href')
                if href: 
                    # Handle relative URLs
                    if href.startswith('/'):
                        full_url = BASE_URL + href
                    elif href.startswith(BASE_URL):
                        full_url = href
                    else:
                        continue # Skip external links

                    # Check if the URL contains any of our valid keywords
                    if any(pattern in full_url for pattern in valid_patterns):
                        # Avoid self-reference (adding the current page back to the queue)
                        if full_url != url:
                            found_urls.append(full_url)

            return list(set(found_urls))

    except Exception as e:
        print(f"  Error processing {url}: {e}")
        return None

# --- Main Execution ---

def main():
    archive_page_url = f"{BASE_URL}/archivos"

    print("--- PHASE 1: DISCOVERING CATEGORIES ---")
    category_urls = get_all_category_urls(archive_page_url)
    
    all_initial_recipe_urls = []
    print("\n--- PHASE 2: DISCOVERING INITIAL RECIPES ---")
    for cat_url in category_urls:
        all_initial_recipe_urls.extend(get_recipe_urls_from_category(cat_url))
        time.sleep(1)  # Be polite with server requests

    # Initialize Queue and History Set
    # We convert the list to a set first to remove initial duplicates.
    urls_to_scrape = deque(list(set(all_initial_recipe_urls)))
    scraped_urls = set()
    all_recipes = []

    print(f"\n--- PHASE 3: CRAWLING {len(urls_to_scrape)} URLs ---")

    # Keep running as long as there are URLs in the queue
    while urls_to_scrape:
        current_url = urls_to_scrape.popleft()

        if current_url in scraped_urls:
            continue
        # Mark as visited
        scraped_urls.add(current_url)

        result = scrape_directoalpaladar_page(current_url)

        # Handle the result
        if isinstance(result, dict):
            all_recipes.append(result)
            print(f"  [SAVED] {result.get('title', 'No Title')}")
        elif isinstance(result, list):
            for new_url in result:
                if new_url not in scraped_urls:
                    urls_to_scrape.append(new_url)
        
        time.sleep(1)

    print(f"\n--- PHASE 4: SAVING {len(all_recipes)} RECIPES ---")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_recipes, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()