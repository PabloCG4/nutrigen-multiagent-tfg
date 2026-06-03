import os
import torch
from torchvision import transforms
import chromadb
from pathlib import Path

from src.vision import config
from src.vision.processing.transforms import SquarePadding

class SemanticEngine:
    """
    The core recognition engine. It acts as a bridge between the raw image crops 
    found by the detector and the vector database to find the exact product identity.
    """

    def __init__(self, collection_name="product_embeddings", computation_device="cuda"):
        self.active_device = computation_device
        
        print(" Loading raw DINOv2 model (Baseline mode)...")
        # Load the base Vision Transformer. This acts as our feature extractor.
        self.vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.vision_encoder.to(self.active_device)
        self.vision_encoder.eval() 

        # Define the exact same preprocessing pipeline used when populating the database
        # to ensure the visual representations are mathematically comparable.
        self.preprocessing_pipeline = transforms.Compose([
            SquarePadding(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Connect to the offline Vector Database
        database_path_string = str(Path(config.DATA_DIR) / "chroma_db_store" / "chroma_db_raw_baseline")
        print(f" Connecting to ChromaDB at: {database_path_string}")
        
        self.vector_database_client = chromadb.PersistentClient(path=database_path_string)
        self.embedding_collection = self.vector_database_client.get_collection(
            name=collection_name
        )

    def extract_batch_embeddings(self, image_crops_list, batch_size=16):
        """
        Converts a list of image crops into a list of mathematical vectors (embeddings).
        Refactored to use list comprehensions for cleaner and more Pythonic execution.
        """
        extracted_embeddings_list = []
        total_image_crops = len(image_crops_list)

        # Disable gradient tracking to save memory and speed up inference
        with torch.no_grad():
            for current_start_index in range(0, total_image_crops, batch_size):
                
                # Slice the list to get the current batch
                current_end_index = min(current_start_index + batch_size, total_image_crops)
                current_batch_images = image_crops_list[current_start_index:current_end_index]

                # Apply transformations to all images in the batch efficiently
                processed_tensors = [
                    self.preprocessing_pipeline(single_crop) 
                    for single_crop in current_batch_images
                ]
    
                batched_input_tensor = torch.stack(processed_tensors).to(self.active_device)

                # Extract visual features using DINOv2
                raw_feature_vectors = self.vision_encoder(batched_input_tensor)
                
                # Normalize the vectors (L2) so they can be compared using Cosine Similarity
                l2_normalized_vectors = torch.nn.functional.normalize(raw_feature_vectors, p=2, dim=1)
                
                # Move back to CPU and convert to a standard Python list. This is necessary for compatibility with ChromaDB's API, which expects native Python data structures.
                numpy_embeddings_array = l2_normalized_vectors.cpu().numpy()
                extracted_embeddings_list.extend(numpy_embeddings_array)

        return extracted_embeddings_list

    def convert_distance_to_confidence(self, cosine_distance_value):
        """
        Transforms the mathematical distance metric into an intuitive 0-100% confidence score.
        Refactored to use max() for cleaner bounds checking.
        """
        similarity_score = 1.0 - cosine_distance_value
        confidence_percentage = similarity_score * 100.0
        
        # Ensure the percentage never drops below 0%
        return max(0.0, round(confidence_percentage, 2))

    def evaluate_candidates(self, detected_candidates_data, maximum_distance_threshold=0.55, top_matches_to_return=50, valid_barcodes_filter=None):
        """
        Takes the unknown object crops found by YOLO, extracts their visual fingerprint, 
        and searches the database for the most likely product matches.
        """
        if not detected_candidates_data:
            return []

        # In candidate dictionaries, we have this structure: {'box': [x_min, y_min, x_max, y_max], 'crop': PIL.Image, 'confidence': float}
        candidate_crop_images = [candidate['crop'] for candidate in detected_candidates_data]
        
        print(f" Extracting raw embeddings for {len(candidate_crop_images)} candidates...")
        query_embeddings = self.extract_batch_embeddings(candidate_crop_images)

        validated_products_list = []

        # Prepare the search parameters for the vector database
        database_query_arguments = {
            "query_embeddings": query_embeddings,
            "n_results": top_matches_to_return
        }

        # Apply a pre-filter if we only want to search within a specific subset of products
        if valid_barcodes_filter and len(valid_barcodes_filter) > 0:
            # "where" used for server-side filtering in ChromaDB. 
            # "$in" operator checks if the barcode in the database matches any in our provided set, which is much faster than client-side filtering after retrieving results.
            database_query_arguments["where"] = {"barcode": {"$in": list(valid_barcodes_filter)}}

        print(" Querying Vector Database...")
        database_query_results = self.embedding_collection.query(**database_query_arguments)

        # The database returns a dictionary with 'ids', 'distances', and 'metadatas' for each query embedding.
        query_distances_matrix = database_query_results.get('distances', [])
        query_metadata_matrix = database_query_results.get('metadatas', [])

        # Process the search results and format the output
        for candidate_index in range(len(detected_candidates_data)):
            current_candidate_info = detected_candidates_data[candidate_index]
            
            # Safety check: ensure the database returned valid results for this specific candidate
            # >= Because the index starts at 0, and the length at 1, so if nothing went wrong, candidate_index should be length-1 at most.
            if candidate_index >= len(query_distances_matrix) or len(query_distances_matrix[candidate_index]) == 0:
                continue
            
            # The database returns results sorted by relevance, so index 0 is the best match
            best_semantic_distance = query_distances_matrix[candidate_index][0]
            
            # Only consider this candidate if the best match is visually similar enough
            if best_semantic_distance <= maximum_distance_threshold:

                # Deduplicate by barcode: the database may now contain several
                # images per product (stored with unique IDs like "{barcode}_{i}").
                # Keep only the single best (lowest) distance per barcode so that
                # downstream stages see one clean entry per product identity.
                barcode_best = {}
                number_of_returned_matches = len(query_distances_matrix[candidate_index])
                for rank_index in range(number_of_returned_matches):
                    specific_match_distance = query_distances_matrix[candidate_index][rank_index]
                    specific_match_metadata = query_metadata_matrix[candidate_index][rank_index]
                    barcode = specific_match_metadata['barcode']
                    if barcode not in barcode_best or specific_match_distance < barcode_best[barcode][0]:
                        barcode_best[barcode] = (specific_match_distance, specific_match_metadata)

                # Sort deduplicated results by distance ascending (best match first)
                sorted_barcode_matches = sorted(
                    barcode_best.items(),
                    key=lambda item: item[1][0]
                )

                candidate_top_matches_list = []
                for rank_index, (barcode, (dist, meta)) in enumerate(sorted_barcode_matches):
                    match_confidence_percentage = self.convert_distance_to_confidence(dist)
                    match_details_dictionary = {
                        'rank': rank_index + 1,
                        'barcode': barcode,
                        'semantic_distance': dist,
                        'confidence_percentage': match_confidence_percentage,
                        'image_path': meta.get('path', '')
                    }
                    candidate_top_matches_list.append(match_details_dictionary)

                # Combine the original YOLO bounding box data with our new semantic identification
                validated_candidate_payload = {
                    'box': current_candidate_info['box'],
                    'yolo_confidence': current_candidate_info.get('yolo_confidence', current_candidate_info.get('confidence', 0.0)),
                    'best_semantic_distance': best_semantic_distance,
                    # Here we have the top N matches for this candidate, each with its own confidence score and metadata, which can be used for further processing.
                    'top_matches': candidate_top_matches_list 
                }
                validated_products_list.append(validated_candidate_payload)

        # This number should be the number of crops that YOLO identified as potential products and passed to the semantic engine for evaluation.
        total_candidates = len(detected_candidates_data)
        # This number should be the number of crops that passed the semantic similarity threshold and are considered valid product identifications.
        total_valid = len(validated_products_list)
        print(f" Semantic filtering complete. Valid products: {total_valid} out of {total_candidates}")
        
        return validated_products_list