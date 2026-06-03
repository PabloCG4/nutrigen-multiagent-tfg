import os
from pathlib import Path

# --- 1. PROJECT & PATHS ---
# Project Root (Up 3 levels: config.py -> text_engine -> src -> TFG)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data", "text")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MASTER_DATA_DIR = os.path.join(DATA_DIR, "master")
RECIPENLG_DIR = os.path.join(BASE_DIR, "RecipeNLG") 

# Specific File Paths
MASTER_RECIPE_PATH = os.path.join(MASTER_DATA_DIR, "master_recipe_dataset_enriched.csv")
RECIPENLG_RAW_PATH = os.path.join(RECIPENLG_DIR, "RecipeNLG_dataset.csv")

# --- 2. RAG CONFIGURATION ---
CHROMA_DB_DIR = "C:/chroma_db_store_enriched_backup"
RAG_COLLECTION_NAME = "recipes_v1"
RAG_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RAG_BATCH_SIZE = 1000

# --- 3. DATA PROCESSING (ETL) ---

# Standard Output Schema
OUTPUT_COLUMNS = ['title', 'link', 'source', 'final_ingredients', 'structured_directions']

# Scraped Files to Process
SCRAPED_FILES_TO_PROCESS = [
    "scraped_directoalpaladar_COMPLETO_v2.json", 
    "myprotein_recipes_ALL.json",
    "scraped_airfryer.json",
    "scraped_ninja.json"
]

# --- 4. PROCESSED FILES PATHS (Inputs for Unification) ---
# We define these explicitly to ensure Unification script picks up exactly what Processing script outputs
PROCESSED_RECIPENLG_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_RecipeNLG.csv")
PROCESSED_PALADAR_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_scraped_directoalpaladar_COMPLETO_v2.csv")
PROCESSED_MYPROTEIN_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_myprotein_recipes_ALL.csv")
PROCESSED_AIRFRYER_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_scraped_airfryer.csv")
PROCESSED_NINJA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_scraped_ninja.csv")

# --- 5. REPORTING & FIGURES ---
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")