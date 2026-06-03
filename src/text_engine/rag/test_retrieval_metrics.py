import time
import statistics
import platform
import torch
import math
from typing import List, Dict
from src.text_engine.rag.retrieval_service import get_recipe_retriever

class RetrievalBenchmark:
    """
    Professional benchmarking suite for the RAG Retrieval System.
    Evaluates:
    1. System Latency (Inference time).
    2. Precision@K (Accuracy).
    3. Mean Reciprocal Rank (MRR - Ranking Quality for first hit).
    4. NDCG (Normalized Discounted Cumulative Gain - Global Ranking Quality).
    """

    def __init__(self):
        print("[INFO] Initializing Benchmark Suite...")
        self.search_engine = get_recipe_retriever()
        self.execution_logs: List[Dict] = []
        
        self.system_info = {
            "OS": platform.system(),
            "Processor": platform.processor(),
            "GPU_Available": torch.cuda.is_available(),
            "Device_Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

    def calculate_reciprocal_rank(self, relevant_positions: List[int]) -> float:
        """
        Calculates MRR. (1 / rank of first relevant item).
        """
        if not relevant_positions:
            return 0.0
        
        best_rank = min(relevant_positions)
        return 1.0 / best_rank

    def calculate_ndcg(self, relevant_positions: List[int], k: int) -> float:
        """
        Calculates NDCG (Normalized Discounted Cumulative Gain).
        Standard metric for ranking quality in Information Retrieval.
        """
        if not relevant_positions:
            return 0.0

        # 1. Calculate DCG (Discounted Cumulative Gain) for our results
        # Formula: sum(rel_i / log2(i + 1))
        dcg = 0.0
        for i in range(1, k + 1):
            if i in relevant_positions:
                dcg += 1.0 / math.log2(i + 1)

        # 2. Calculate IDCG (Ideal DCG)
        # This is the score if all relevant items were at the very top (positions 1, 2, 3...)
        num_relevant = len(relevant_positions)
        idcg = 0.0
        for i in range(1, num_relevant + 1):
            idcg += 1.0 / math.log2(i + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def generate_performance_report(self, latencies: List[float]):
        """
        Generates statistical summary.
        """
        if not self.execution_logs:
            print("[WARN] No data available to report.")
            return

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        precision_scores = [log['precision'] for log in self.execution_logs]
        mrr_scores = [log['reciprocal_rank'] for log in self.execution_logs]
        ndcg_scores = [log['ndcg'] for log in self.execution_logs]
        
        mean_precision = statistics.mean(precision_scores) * 100
        mean_mrr = statistics.mean(mrr_scores)
        mean_ndcg = statistics.mean(ndcg_scores)

        print("FINAL ENGINEERING REPORT")
        print(f"Hardware: {self.system_info['Device_Name']}")
        print("PERFORMANCE (LATENCY):")
        print(f"Average Inference Time: {avg_latency:.2f} ms")
        print(f"Max Inference Time:     {max_latency:.2f} ms")
        print("QUALITY METRICS:")
        print(f"Mean Precision: {mean_precision:.2f}%")
        print(f"Mean MRR:       {mean_mrr:.4f} (First Hit Position)")
        print(f"Mean NDCG:      {mean_ndcg:.4f} (Global Ranking Quality)")

    def run_audit(self, query_list: List[str], top_k: int = 3):
        """
        Executes the Human-In-The-Loop (HITL) audit process.
        """
        print(f"\n[START] Starting Audit on {self.system_info['Device_Name']}")
        print(f"Configuration: Top-K = {top_k}")
        
        latencies = []

        for index, query in enumerate(query_list, 1):
            print(f"\n[TEST {index}/{len(query_list)}] Query: '{query}'")
            
            # Measure Latency
            start_time = time.perf_counter()
            results = self.search_engine.search_recipes(query, top_k=top_k)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if not results:
                print("[INFO] No results found.")
                self.execution_logs.append({
                    "query": query, "latency_ms": latency_ms,
                    "precision": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0
                })
                continue

            # Display Candidates
            print(f"Latency: {latency_ms:.2f} ms")
            for i, recipe in enumerate(results, 1):
                print(f"   [{i}] {recipe['title']} (Score: {recipe['final_score']:.4f})")
                print(f"       Ingredients: {recipe['ingredients']}")

            # User Input
            valid_input = False
            relevant_ranks = []
            while not valid_input:
                try:
                    user_input = input("Enter numbers of relevant results (e.g., '1 3' or '0'): ")
                    if user_input.strip() == '0':
                        relevant_ranks = []
                    else:
                        relevant_ranks = [int(x) for x in user_input.split()]
                    valid_input = True
                except ValueError:
                    print("[ERROR] Invalid input. Use numbers separated by space.")

            # Compute Metrics
            relevant_count = len(relevant_ranks)
            precision = relevant_count / top_k
            reciprocal_rank = self.calculate_reciprocal_rank(relevant_ranks)
            ndcg = self.calculate_ndcg(relevant_ranks, top_k)

            self.execution_logs.append({
                "query": query,
                "latency_ms": latency_ms,
                "precision": precision,
                "reciprocal_rank": reciprocal_rank,
                "ndcg": ndcg
            })

        self.generate_performance_report(latencies)

if __name__ == "__main__":
    test_queries = [
        "Arroz, pollo, huevo",                
        "Lentejas, cebolla, tomate",  
        "Galletas, leche, azúcar",    
        "Macarrones, queso, tomate", 
        "Patata, cebolla, huevo"       
    ]
    
    benchmark_tool = RetrievalBenchmark()
    benchmark_tool.run_audit(test_queries, top_k=3)