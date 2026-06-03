import chromadb
from regex import F
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
import sys
import nltk
import time
from nltk.corpus import stopwords
from src.text_engine import config


_is_warmed_up: bool = False


class RecipeRetriever:
    def __init__(self):
        """
        Initializes the Retriever service.
        Connects to the existing Vector Database and loads the Embedding Model.
        """
        print("--- Initializing Retrieval Service ---")
        
        # Setup NLTK (Stopwords)
        nltk.data.find('corpora/stopwords')
        self.stop_words = set(stopwords.words('english') + stopwords.words('spanish'))

        db_path = config.CHROMA_DB_DIR
        self.client = chromadb.PersistentClient(path=db_path)
        # Connect to Collection
        self.collection = self.client.get_collection(name=config.RAG_COLLECTION_NAME)
        # Load Model
        print(f"Loading Model: {config.RAG_EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.RAG_EMBEDDING_MODEL)
        
        # Avoid calling `collection.count()` here: on large persistent stores it can be slow
        # and may force loading index segments during startup.
        print("Service Ready. Collection loaded.")

        # Warm-up: first vector query often pays HNSW/disk cold-start cost; do it here so
        # the Chef's first real search is not penalized as badly.
        try:
            t_w0 = time.perf_counter()
            wvec = self.model.encode("warmup").tolist()
            self.collection.query(
                query_embeddings=[wvec],
                n_results=1,
                include=["metadatas", "distances"],
            )
            print(f"[RecipeRetriever] Warm-up query: {(time.perf_counter() - t_w0):.3f}s")
        except Exception as exc:
            print(f"[RecipeRetriever] Warm-up query skipped: {exc}")

        global _is_warmed_up
        if _is_warmed_up:
            return

        warmup_enabled = (os.getenv("CHROMA_WARMUP_ENABLED") or "").strip()
        if warmup_enabled != "1":
            return

        # Warm-up with representative `where` filters.
        # This targets the same slow path Chef uses (dish_type + ingredient_count), so
        # the FIRST real request in a fresh Celery worker is less likely to pay a 400–500s penalty.
        warmup_wheres: list[dict[str, Any]] = [
            {"$and": [{"dish_type": {"$eq": "main_course"}}, {"ingredient_count": {"$lte": 8}}]},
            {"$and": [{"dish_type": {"$eq": "salad_side"}}, {"ingredient_count": {"$lte": 6}}]},
            {"ingredient_count": {"$lte": 10}},
        ]

        try:
            for idx, where in enumerate(warmup_wheres, start=1):
                t0 = time.perf_counter()
                # reuse the same embedding to minimize encode overhead during warm-up
                self.collection.query(
                    query_embeddings=[wvec],
                    n_results=1,
                    include=["metadatas", "distances"],
                    where=where,
                )
                dt = time.perf_counter() - t0
                print(f"[RecipeRetriever] Warm-up where query #{idx}: {dt:.3f}s | where={where}")
            _is_warmed_up = True
        except Exception as exc:
            print(f"[RecipeRetriever] Warm-up where queries skipped: {exc}")

    def refine_ranking(self, query: str, results: dict, final_k: int) -> List[Dict[str, Any]]:
        """
        Applies custom weighting logic using NLTK for linguistic filtering.
        """
        candidates = []
        query_lower = query.lower()
        
        # Remove stop words
        important_words = []
        for w in query_lower.split():
            if w not in self.stop_words:
                important_words.append(w)

        # If the user only typed stop words, keep original to avoid empty list
        if not important_words:
            important_words = query_lower.split()

        num_results = len(results['ids'][0])
        
        for i in range(num_results):
            metadata = results['metadatas'][0][i]
            original_dist = results['distances'][0][i] 
            final_score = original_dist
            title = metadata.get('title', '').lower()
            ingredients = metadata.get('ingredients', '').lower()

            # 1. KEYWORD MATCHING
            matches_title = 0
            matches_ingredients = 0

            for word in important_words:
                if word in ingredients:
                    matches_ingredients += 1
                elif word in title:
                    matches_title += 1

            # Apply boosts (lower distance is better):
            # - Ingredient coverage dominates ranking.
            # - Title matches are treated as a mild bonus only.
            if matches_ingredients > 0:
                # Distance reduction for ingredient matches.
                final_score *= (0.75 ** matches_ingredients)
            if matches_title > 0:
                final_score *= (0.95 ** matches_title)

            candidates.append({
                "title": metadata.get('title', 'Unknown'),
                "ingredients": metadata.get('ingredients', 'Unknown'),
                "original_distance": original_dist,
                "final_score": final_score
            })
            
        # Sort by Final Score (Lower is better)
        candidates.sort(key=lambda x: x['final_score'])
        
        return candidates[:final_k]

    def search_recipes(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a semantic search in the vector database with Title boosting.
        """
        # Convert User Query to Vector
        query_vector = self.model.encode(query).tolist()
        
        # Perform Similarity Search in ChromaDB
        FETCH_POOL_SIZE = 2
        fetch_k = top_k * FETCH_POOL_SIZE  # Fetch more than needed to allow for re-ranking
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []

        # Refine Ranking with Custom Logic
        final_results = self.refine_ranking(query, results, top_k)
        
        return final_results


_recipe_retriever_singleton: Optional["RecipeRetriever"] = None


def get_recipe_retriever() -> "RecipeRetriever":
    """
    Single shared RecipeRetriever for the process (one Chroma PersistentClient + encoder).
    Avoids opening multiple clients against the same CHROMA_DB_DIR from Chef tools and RAG tools.
    """
    global _recipe_retriever_singleton
    if _recipe_retriever_singleton is None:
        _recipe_retriever_singleton = RecipeRetriever()
    return _recipe_retriever_singleton


# --- Test Execution Block ---
if __name__ == "__main__":
    try:
        retriever = get_recipe_retriever()
        
        while True:
            user_query = input("Enter a search query (or 'e' to exit): ")
            if user_query.lower() == 'e':
                break
                
            # We ask for the top results, but the system internally fetches more and picks the best based on title
            results = retriever.search_recipes(user_query, top_k=5)

            print(f"\n--- Top Results for '{user_query}' ---")
            for idx, res in enumerate(results):
                print(f"\n{idx+1}. {res['title']}")
                print(f"   Source: {res['source']}")
                print(f"   Ingredients snippet: {res['ingredients']}")
                print(f"   Orig. Dist: {res['original_distance']} | Boosted Score: {res['final_score']}")
                
    except Exception as e:
        print(f"Error: {e}")