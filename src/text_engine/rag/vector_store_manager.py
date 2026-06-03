import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys
import time
import torch
from src.text_engine import config

class VectorStoreManager:
    def __init__(self):
        """
        Initializes the ChromaDB client and loads the Embedding Model.
        """
        print("--- Initializing Vector Store Manager ---")
        
        # 1. Setup Persistent Client — enriched DB lives in a separate directory so
        #    the original index is never overwritten.
        db_path = config.CHROMA_DB_DIR
        print(f"Database Path: {db_path}")

        if os.path.exists(db_path):
            print("INFO: Loading existing enriched database...")
        else:
            print("INFO: Creating new enriched database storage...")

        self.client = chromadb.PersistentClient(path=db_path)
        
        # 2. Load the Embedding Model with GPU Acceleration
        device = 'cuda'

        self.model = SentenceTransformer(config.RAG_EMBEDDING_MODEL, device=device)
        
        # 3. Get or Create the Collection
        self.collection = self.client.get_or_create_collection(
            name=config.RAG_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity (0 to 1 score) for better performance with text embeddings
        )
        print("System ready for ingestion.")

    def _prepare_text_for_embedding(self, row) -> str:
        """
        Creates the unique text string that represents the recipe in vector space.
        Strategy: Combine Title + Ingredients.
        Instructions are excluded from the vector to reduce noise but kept in metadata.
        """
        title = str(row['title']).strip()
        ingredients = str(row['final_ingredients']).strip()
        
        # Optimized format for semantic search
        return f"RECIPE: {title}. INGREDIENTS: {ingredients}."

    def process_and_ingest(self):
        """
        Reads the Master CSV in chunks and upserts vectors into ChromaDB.
        """
        # Read the enriched dataset produced by enrich_dataset.py
        csv_path = config.MASTER_RECIPE_PATH

        if not os.path.exists(csv_path):
            print(f"CRITICAL ERROR: Master dataset not found at {csv_path}")
            return

        print(f"Starting ingestion from: {csv_path}")
        print(f"Batch Size: {config.RAG_BATCH_SIZE}")

        # Read CSV using a generator (chunksize) to prevent RAM saturation
        chunk_iterator = pd.read_csv(csv_path, chunksize=config.RAG_BATCH_SIZE)
        
        total_records_processed = 0
        
        for chunk in tqdm(chunk_iterator, desc="Processing and Indexing"):
            
            documents = []  # Text to be vectorized
            metadatas = []  # Payload data for retrieval
            ids = []        # Unique ID
            
            for idx, row in chunk.iterrows():
                # Minimal validation
                if pd.isna(row['title']):
                    continue
                
                # Prepare text for embedding
                vector_text = self._prepare_text_for_embedding(row)
                
                documents.append(vector_text)
                ids.append(str(idx)) # Use dataframe index as ID
                
                # Store FULL metadata so the Agent is self-sufficient.
                # dish_type and ingredient_count come from the enrichment pass.
                metadatas.append({
                    "title":            str(row['title']),
                    "source":           str(row.get('source', 'Unknown')),
                    "url":              str(row.get('link', '')),
                    "ingredients":      str(row['final_ingredients']),
                    "instructions":     str(row.get('structured_directions', '')),
                    "dish_type":        str(row.get('dish_type', 'main_course')),
                    "ingredient_count": int(row.get('ingredient_count', 0)),
                })

            if not documents:
                continue

            # --- HEAVY LIFTING: Generate Embeddings ---
            # GPU acceleration happens automatically here if device='cuda' was set in init
            embeddings = self.model.encode(documents).tolist()
            
            # --- I/O: Save to Disk ---
            # Retry loop: up to 5 attempts with exponential back-off.
            # If every attempt fails the exception is re-raised to abort the run;
            # skipping a failed batch is not acceptable — no recipe data must be lost.
            _MAX_ATTEMPTS = 5
            last_exc: Exception | None = None
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                try:
                    self.collection.upsert(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids,
                    )
                    break  # Upsert succeeded — proceed to the next batch.
                except Exception as exc:
                    last_exc = exc
                    wait_seconds = 2 ** (attempt - 1)  # 1 s, 2 s, 4 s, 8 s, 16 s
                    if attempt < _MAX_ATTEMPTS:
                        print(
                            f"\n[WARN] Upsert failed on attempt {attempt}/{_MAX_ATTEMPTS} "
                            f"(batch starting at id={ids[0]}). "
                            f"Retrying in {wait_seconds}s... Error: {exc}"
                        )
                        time.sleep(wait_seconds)
                    else:
                        print(
                            f"\n[CRITICAL] Upsert failed after {_MAX_ATTEMPTS} attempts "
                            f"(batch starting at id={ids[0]}). Aborting ingestion."
                        )
                        raise last_exc
            total_records_processed += len(documents)

        print(f"--- Process Complete. {total_records_processed} recipes indexed. ---")

if __name__ == "__main__":
    # Direct execution block
    manager = VectorStoreManager()
    
    # Safety check before starting a long process
    confirmation = input("Do you want to start massive data ingestion? (y/n): ")
    if confirmation.lower() == 'y':
        manager.process_and_ingest()
    else:
        print("Operation cancelled.")