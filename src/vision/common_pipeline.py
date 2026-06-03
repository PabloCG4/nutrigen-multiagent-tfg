import cv2
import numpy as np
import time
import easyocr
import re
import nltk
from collections import Counter
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import socket
import json
from nltk.corpus import stopwords, words

from mobile_sam import sam_model_registry, SamPredictor

# NLTK Spanish/English stoplists include words that are still informative on food labels
# (e.g. "sin azúcar", "con sal"). These are removed from the stopword set used in OCR sanitization.
_OCR_RETAIN_DESPITE_STOPWORD = frozenset({
    "sin",
    "con",
})
from src.vision import config
from src.vision.processing.semantic_engine import SemanticEngine

class BaseSmartFridgeSystem:
    """
    Abstract Base Class integrating Phase 1 Detection
    with a Semantic Engine (DINOv2 + ChromaDB) for Zero-Shot classification.
    Specific domain logic (photometrics, multiscale detection, and spatial deduplication)
    must be implemented by child classes following the Strategy Pattern.
    """

    def __init__(self, yolo_path, semantic_distance_threshold, device=None):
        """
        Initializes the complete semantic system.
        Detection initialization and thresholds are delegated to the child classes.
        """
        self.computation_device = device if device else "cuda"
        self.yolo_path = yolo_path
        self.semantic_distance_threshold = semantic_distance_threshold
        self.ocr_reader = easyocr.Reader(['es', 'en'], gpu=True)

        print("Loading MobileSAM for background removal...")
        sam_checkpoint_path = Path(config.SAM_MODEL_DIR) / "mobile_sam.pt"

        if not Path(sam_checkpoint_path).exists():
            print(f"SAM checkpoint not found at {sam_checkpoint_path}. Background removal bypassed.")
            self.sam_predictor = None
        else:
            mobile_sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint_path)
            mobile_sam.to(device=self.computation_device)
            self.sam_predictor = SamPredictor(mobile_sam)

        print("Loading Semantic Engine...")
        self.semantic_engine = SemanticEngine()

        text_database_path = Path(config.DATA_DIR) / "product_names_mapping.json"
        self.product_text_database = self.load_product_text_database(text_database_path)

        self.build_sparse_index()
        self.setup_nlp_filters()

    def build_sparse_index(self):
        """
        Builds the BM25 inverted index in memory using the product text database.
        Extracts a frequency-sorted vocabulary list to resolve Levenshtein ties
        (e.g., forcing 'grizgo' -> 'griego' instead of 'grigio').
        """
        self.sparse_barcodes_list = list(self.product_text_database.keys())
        corpus_names = list(self.product_text_database.values())

        tokenized_corpus = []
        word_frequencies = Counter()

        for name in corpus_names:
            cleaned_name = re.sub(r'[^a-záéíóúüñ0-9\s%]', '', name.lower()).strip()
            tokens = cleaned_name.split()
            tokenized_corpus.append(tokens)
            word_frequencies.update(tokens)

        self.sparse_retrieval_engine = BM25Okapi(tokenized_corpus)
        self.valid_vocabulary_set = set(word_frequencies.keys())

        def get_word_frequency(word):
            return word_frequencies[word]

        self.valid_vocabulary_list = sorted(
            list(self.valid_vocabulary_set),
            key=get_word_frequency,
            reverse=True
        )

        print(f"BM25 index built: {len(self.valid_vocabulary_set)} vocabulary words.")

    def execute_sparse_retrieval(self, extracted_text, top_k_results=50):
        """
        Queries the BM25 engine with the sanitized OCR text to retrieve candidate barcodes.
        Confidence is scaled proportionally to the BM25 score (5–30% range) so that
        stronger text matches carry more signal into the late fusion step.
        """
        if not extracted_text:
            return []

        tokenized_query = extracted_text.split()
        document_scores = self.sparse_retrieval_engine.get_scores(tokenized_query)

        # Get the top k results from the document scores.
        top_indices = np.argsort(document_scores)[::-1][:top_k_results]

        # Find the maximum positive score for proportional normalization.
        # A fixed 30% for all BM25 matches discards relative quality information.
        max_bm25_score = 0.0
        for index in top_indices:
            if document_scores[index] > max_bm25_score:
                max_bm25_score = document_scores[index]

        sparse_matches = []
        # Iterate through the top k results and add them to the sparse matches list.
        for rank, index in enumerate(top_indices):
            score = document_scores[index]
            if score <= 0:
                continue

            # Scale to a 5–30% range relative to the best match in the batch
            if max_bm25_score > 0:
                normalized = score / max_bm25_score
                confidence = max(5.0, normalized * 30.0)
            else:
                confidence = 5.0

            barcode = self.sparse_barcodes_list[index]
            sparse_matches.append({
                'rank': rank + 1,
                'barcode': barcode,
                'semantic_distance': 1.0,
                'confidence_percentage': round(confidence, 2),
                'image_path': '',
                'source': 'BM25'
            })

        return sparse_matches

    def fuse_hybrid_results(self, dense_products, extracted_candidates):
        """
        Merges dense (DINOv2) and sparse (BM25) retrieval results.
        Injects a Dynamic Consensus Bonus based on OCR-to-Product similarity,
        scaling the max +12.0 bonus by the text match percentage.
        """
        fused_products_list = []

        # Build a lookup dict: box_tuple -> dense_product for O(1) access
        dense_products_dictionary = {}
        for product in dense_products:
            box_key = tuple(product['box'])
            dense_products_dictionary[box_key] = product

        for candidate in extracted_candidates:
            current_box = candidate['box']
            extracted_text = candidate.get('extracted_text', '').strip()
            merged_matches_dictionary = {}

            dense_match = dense_products_dictionary.get(tuple(current_box))

            if dense_match:
                for match in dense_match['top_matches']:
                    match['source'] = 'DINOv2'
                    merged_matches_dictionary[match['barcode']] = match

            sparse_matches = self.execute_sparse_retrieval(extracted_text)

            for match in sparse_matches:
                barcode = match['barcode']
                if barcode in merged_matches_dictionary:
                    merged_matches_dictionary[barcode]['source'] = 'DINOv2 + BM25'
                    base_confidence = merged_matches_dictionary[barcode]['confidence_percentage']

                    product_name = self.product_text_database.get(barcode, "").lower().strip()
                    if extracted_text and product_name:
                        text_sim = fuzz.token_set_ratio(extracted_text, product_name)
                        dynamic_bonus = (text_sim / 100.0) * 12.0
                    else:
                        dynamic_bonus = 0.0

                    merged_matches_dictionary[barcode]['confidence_percentage'] = min(base_confidence + dynamic_bonus, 99.9)
                else:
                    merged_matches_dictionary[barcode] = match

            if not merged_matches_dictionary:
                continue

            def get_match_confidence(match):
                return match['confidence_percentage']

            sorted_merged_matches = sorted(
                list(merged_matches_dictionary.values()),
                key=get_match_confidence,
                reverse=True
            )

            for rank_index, match in enumerate(sorted_merged_matches):
                match['rank'] = rank_index + 1

            fused_payload = {
                'box': current_box,
                'yolo_confidence': candidate['yolo_confidence'],
                'best_semantic_distance': sorted_merged_matches[0]['semantic_distance'],
                'top_matches': sorted_merged_matches,
                'extracted_text': extracted_text,
                'raw_ocr': candidate.get('raw_ocr', '')
            }
            fused_products_list.append(fused_payload)

        return fused_products_list

    def load_product_text_database(self, database_filepath):
        """
        Loads the barcode-to-name mapping into memory for global OCR search.
        Expects a JSON file formatted as a flat dictionary.
        """
        file_path = Path(database_filepath)

        if not file_path.exists():
            print(f"Product text database not found at {file_path}")
            return {}

        # Load the JSON file into a dictionary, each barcode is the key and the value is the product name, for easy access.
        json_file = open(file_path, 'r', encoding='utf-8')
        product_dictionary = json.load(json_file)
        json_file.close()

        print(f"Product database loaded: {len(product_dictionary)} records.")
        return product_dictionary

    def calculate_intersection_over_minimum(self, box_a, box_b):
        """
        Calculates the Intersection over Minimum Area (IoM).
        This is superior to IoU for detecting nested objects (like a logo inside a bottle)
        as it identifies if one box is contained within another.
        """
        x_min_intersection = max(box_a[0], box_b[0])
        y_min_intersection = max(box_a[1], box_b[1])
        x_max_intersection = min(box_a[2], box_b[2])
        y_max_intersection = min(box_a[3], box_b[3])

        intersection_width = max(0, x_max_intersection - x_min_intersection)
        intersection_height = max(0, y_max_intersection - y_min_intersection)
        intersection_area = intersection_width * intersection_height

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        minimum_area = min(area_a, area_b)

        if minimum_area == 0:
            return 0.0

        return intersection_area / minimum_area

    def calculate_cross_validation_scores(self, valid_products):
        """
        Calculates a combined weighted score using both YOLO and semantic confidences.
        """
        if not valid_products:
            return []

        # Semantic confidence is weighted heavily because it directly reflects the
        # product match quality. YOLO confidence is secondary (detection quality only).
        semantic_weight = 0.85
        yolo_weight = 0.15

        for product in valid_products:
            best_match = product['top_matches'][0]
            normalized_semantic_confidence = best_match['confidence_percentage'] / 100.0
            yolo_confidence = product['yolo_confidence']

            combined_score = (normalized_semantic_confidence * semantic_weight) + (yolo_confidence * yolo_weight)
            product['cross_validation_score'] = combined_score

        def get_cross_val_score(product):
            return product['cross_validation_score']

        sorted_products = sorted(valid_products, key=get_cross_val_score, reverse=True)
        return sorted_products

    def refine_box_with_sam(self, rgb_image_array, box_coordinates):
        """
        Uses MobileSAM in multi-mask mode to solve the Ambiguity Problem.
        Generates 3 granular masks (whole, part, subpart) and strictly selects
        the one that maximizes the area within the YOLO bounding box, effectively
        ignoring high-contrast labels or sub-components.
        """
        x1, y1, x2, y2 = box_coordinates
        original_crop = rgb_image_array[y1:y2, x1:x2]
        if self.sam_predictor is None:
            return box_coordinates, Image.fromarray(original_crop)

        try:
            self.sam_predictor.set_image(rgb_image_array)
            input_bounding_box = np.array(box_coordinates)

            # multimask_output=True forces SAM to return 3 masks representing
            # different hierarchical levels of the object (whole, part, subpart).
            segmentation_masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_bounding_box[None, :],
                multimask_output=True,
            )
            if len(segmentation_masks) == 0:
                return box_coordinates, Image.fromarray(original_crop)

            yolo_area = (x2 - x1) * (y2 - y1)
            best_refined_box = None
            highest_area_ratio = 0.0

            # Select the mask that covers the largest portion of the YOLO box.
            # This naturally discards masks that isolated tiny labels or reflections.
            for mask in segmentation_masks:
                y_indices, x_indices = np.where(mask)
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue

                new_x1, new_x2 = int(np.min(x_indices)), int(np.max(x_indices))
                new_y1, new_y2 = int(np.min(y_indices)), int(np.max(y_indices))
                # Constrain the mask strictly to the YOLO prompt boundaries
                final_x1, final_y1 = max(x1, new_x1), max(y1, new_y1)
                final_x2, final_y2 = min(x2, new_x2), min(y2, new_y2)
                mask_area = (final_x2 - final_x1) * (final_y2 - final_y1)
                area_ratio = mask_area / yolo_area if yolo_area > 0 else 0.0
                if area_ratio > highest_area_ratio:
                    highest_area_ratio = area_ratio
                    best_refined_box = (final_x1, final_y1, final_x2, final_y2)

            # Fallback: if the best mask destroys more than 30% of the original box,
            # SAM has failed to find the true container borders.
            if best_refined_box is None or highest_area_ratio < 0.70:
                return box_coordinates, Image.fromarray(original_crop)

            refined_crop = rgb_image_array[best_refined_box[1]:best_refined_box[3], best_refined_box[0]:best_refined_box[2]]
            return best_refined_box, Image.fromarray(refined_crop)

        except Exception as processing_error:
            print(f"SAM processing failed for box {box_coordinates}: {processing_error}")
            return box_coordinates, Image.fromarray(original_crop)

    def apply_local_ocr_reranking(self, valid_products):
        """
        Surgical OCR Reranking v3:
        1. Accent-agnostic comparison for strict string matching.
        2. Dynamic Length-Based Fuzzy Thresholding to absorb 1-character OCR errors.
        3. Mild progressive penalty for unmatched dictionary words.
        4. Anchor Filter for hallucination penalty.
        5. Non-linear proportional bonuses and dynamic BM25 penalization.
        """

        if not valid_products:
            return valid_products

        translation_table = str.maketrans('áéíóúü', 'aeiouu')

        for product in valid_products:
            cleaned_ocr = product.get('extracted_text', '').lower().strip()

            if not cleaned_ocr:
                continue

            ocr_no_accents = cleaned_ocr.translate(translation_table)
            ocr_tokens_clean = set(ocr_no_accents.split())

            # Iterate through the top matches from semantic engine and apply the surgical OCR reranking.
            # Each match is a dictionary with the following structure:
            # {
            #     'barcode': barcode,
            #     'confidence_percentage': confidence_percentage,
            #     'source': source
            # }
            for match in product['top_matches']:
                barcode = match['barcode']
                product_name = self.product_text_database.get(barcode, "").lower().strip()
                # Get the base confidence of the match.
                base_confidence = match['confidence_percentage']
                match['pre_rerank_conf'] = base_confidence

                # If the product name is not found, set the surgical similarity and add bonus to 0.
                if not product_name:
                    match['surgical_sim'] = 0.0
                    match['add_bonus'] = 0.0
                    continue

                prod_no_accents = product_name.translate(translation_table)
                prod_tokens_clean = set(prod_no_accents.split())

                # Global character-level ratio
                global_ratio = fuzz.ratio(ocr_no_accents, prod_no_accents)

                # Progressive word-based penalty scoring via dynamic fuzzy intersection
                common_tokens = []
                if prod_tokens_clean and ocr_tokens_clean:
                    for ocr_token in ocr_tokens_clean:
                        token_length = len(ocr_token)
                        if token_length <= 3:
                            dynamic_threshold = 100.0
                        elif token_length <= 6:
                            dynamic_threshold = 75.0
                        else:
                            dynamic_threshold = 70.0

                        # extractOne is a function that extracts the best fuzzy match from the product name, function from rapidfuzz.
                        best_fuzzy_match = process.extractOne(ocr_token, list(prod_tokens_clean), scorer=fuzz.ratio)

                        # We only add the token to the common tokens if the fuzzy match is above the dynamic threshold.
                        if best_fuzzy_match and best_fuzzy_match[1] >= dynamic_threshold:
                            common_tokens.append(best_fuzzy_match[0])

                # The common tokens are the tokens that are present in both the product name and the OCR text.
                if common_tokens:
                    # Calculate the length of the common tokens.
                    len_common_chars = 0
                    for token in common_tokens:
                        len_common_chars += len(token)

                    len_ocr_chars = 0
                    for token in ocr_tokens_clean:
                        len_ocr_chars += len(token)

                    # Calculate the OCR coverage, this is the percentage of the OCR text that is covered by the product name.
                    ocr_coverage = min((len_common_chars / max(len_ocr_chars, 1)) * 100.0, 100.0)

                    # Calculate the unmatched product tokens and the unmatched OCR tokens.
                    unmatched_prod_tokens = len(prod_tokens_clean) - len(common_tokens)
                    unmatched_ocr_tokens = len(ocr_tokens_clean) - len(common_tokens)
                    total_unmatched_words = unmatched_prod_tokens + unmatched_ocr_tokens

                    # Calculate the penalty rate, this is the penalty rate for the unmatched tokens, 0.10 for BM25 and 0.05 for DINOv2.
                    current_source = match.get('source', '')
                    penalty_rate = 0.10 if 'BM25' in current_source and 'DINOv2' not in current_source else 0.05
                    penalty_factor = max(1.0 - (total_unmatched_words * penalty_rate), 0.70)
                    # Calculate the token score, this is the score for the OCR coverage multiplied by the penalty factor.
                    token_score = ocr_coverage * penalty_factor
                else:
                    token_score = 0.0

                raw_similarity = min(max(global_ratio, token_score), 100.0)

                # Anchor filter: penalise matches that share no recognizable word with the OCR.
                has_anchor = len(common_tokens) > 0

                surgical_similarity = raw_similarity if has_anchor else (raw_similarity * 0.4)

                # Exponential bonus curve: rewards strong matches more than marginal ones
                if surgical_similarity > 20:
                    similarity_ratio = surgical_similarity / 100.0
                    additive_bonus = 50.0 * (similarity_ratio ** 2)
                else:
                    additive_bonus = 0.0

                source = match.get('source', '')
                if 'BM25' in source and 'DINOv2' not in source:
                    # Mild dynamic reduction for pure-BM25 matches
                    similarity_ratio = surgical_similarity / 100.0
                    dynamic_multiplier = 0.85 + (similarity_ratio * 0.20)
                    additive_bonus *= dynamic_multiplier

                additive_bonus = min(additive_bonus, 50.0)

                match['surgical_sim'] = round(surgical_similarity, 1)
                match['add_bonus'] = round(additive_bonus, 1)
                match['confidence_percentage'] = round(min(base_confidence + additive_bonus, 99.9), 2)

            def get_match_confidence(match):
                return match['confidence_percentage']

            product['top_matches'] = sorted(
                product['top_matches'],
                key=get_match_confidence,
                reverse=True
            )

            for index, match in enumerate(product['top_matches']):
                match['rank'] = index + 1

            product['best_semantic_distance'] = product['top_matches'][0].get('semantic_distance', 0.0)

        return valid_products

    def apply_global_semantic_deduplication(self, products, top_k_check=1):
        """
        Removes duplicate products across the entire image to ensure a clean,
        binary inventory (ingredient present/not present).
        """
        if not products:
            return []

        def get_cross_val_score(product):
            return product['cross_validation_score']

        sorted_products = sorted(products, key=get_cross_val_score, reverse=True)
        unique_global_products = []

        for current_product in sorted_products:
            current_barcodes = set()
            for match in current_product['top_matches'][:top_k_check]:
                current_barcodes.add(match['barcode'])

            is_global_duplicate = False

            for kept_product in unique_global_products:
                kept_barcodes = set()
                for match in kept_product['top_matches'][:top_k_check]:
                    kept_barcodes.add(match['barcode'])

                if not current_barcodes.isdisjoint(kept_barcodes):
                    is_global_duplicate = True
                    break

            if not is_global_duplicate:
                unique_global_products.append(current_product)

        return unique_global_products

    def setup_nlp_filters(self):
        """
        Loads NLTK resources for OCR sanitization:

        - stop_words: frequent function words (ES+EN) dropped early to reduce noise for BM25.
          Some label-specific words are removed from this set via _OCR_RETAIN_DESPITE_STOPWORD.
        - english_vocab: optional fallback in sanitize step 5 to keep plausible English tokens
          that are not in the product-name vocabulary (does not affect Spanish-only tokens).
        """
        socket.setdefaulttimeout(10.0)

        required_resources = ['words', 'stopwords']

        for resource in required_resources:
            try:
                if resource == 'words':
                    nltk.data.find('corpora/words')
                elif resource == 'stopwords':
                    nltk.data.find('corpora/stopwords')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as download_error:
                    print(f"NLTK download failed for '{resource}': {download_error}")

        try:
            combined_stopwords = set(stopwords.words('spanish')).union(set(stopwords.words('english')))
            self.stop_words = combined_stopwords - _OCR_RETAIN_DESPITE_STOPWORD
            self.english_vocab = set()
            for word in words.words():
                self.english_vocab.add(word.lower())
        except Exception as load_error:
            print(f"NLP components could not load, OCR filtering degraded: {load_error}")
            self.stop_words = set()
            self.english_vocab = set()

    def sanitize_and_correct_ocr_text(self, raw_text):
        """
        Executes the explicit cascade sanitization flow while maintaining
        strict contextual strength analysis to prevent noise survival.
        Implements Lookahead Token Reassembly to fix OCR fragmentation (e.g., 'aceit' + 'una').
        1. Noise removal and number purging (preserves nutritional markers).
        2. Stop-word elimination (ES+EN NLTK lists), except tokens in _OCR_RETAIN_DESPITE_STOPWORD
           (e.g. "sin", "con" on food labels).
        3. Spatial Reassembly (Concatenates fragmented tokens against dictionary).
        4. High-confidence dictionary mapping (Exact or Fuzzy).
        5. Fallback validation (NLTK for weak tokens, preservation for strong OOV).
        """

        if not raw_text:
            return ""

        cleaned_raw = re.sub(r'[^a-záéíóúüñ0-9\s%]', '', raw_text.lower()).strip()
        raw_tokens = cleaned_raw.split()

        nutritional_markers = {"0", "00", "0%", "00%", "100%", "100", "50%", "99%", "1l", "1kg", "500g", "250g"}

        # Step 1: Base cleaning and immediate stop-word removal
        base_tokens = []
        for token in raw_tokens:
            if self.stop_words and token in self.stop_words:
                continue

            if token in nutritional_markers:
                base_tokens.append(token)
                continue

            has_alpha = False
            for char in token:
                if char.isalpha():
                    has_alpha = True
                    break

            if has_alpha:
                clean_token = re.sub(r'\d+', '', token)
                if len(clean_token) >= 2:
                    base_tokens.append(clean_token)

        if not base_tokens:
            return ""

        v_set = self.valid_vocabulary_set
        v_list = self.valid_vocabulary_list

        final_tokens = []
        skip_next = False

        # Steps 2 & 3: Lookahead Token Reassembly and Validation
        for index in range(len(base_tokens)):
            if skip_next:
                skip_next = False
                continue

            current_token = base_tokens[index]
            is_short_token = len(current_token) < 3 and current_token not in nutritional_markers

            merged_successfully = False

            # Forward concatenation: triggered if either current or next token is a short fragment
            if index < len(base_tokens) - 1:
                next_token = base_tokens[index + 1]
                next_is_short = len(next_token) < 3 and next_token not in nutritional_markers

                if is_short_token or next_is_short:
                    combined_string = current_token + next_token
                    best_match = process.extractOne(combined_string, v_list, scorer=fuzz.ratio) if v_list else None

                    dynamic_threshold = 100.0 if len(combined_string) <= 3 else 70.0

                    if best_match and best_match[1] >= dynamic_threshold:
                        final_tokens.append(best_match[0])
                        skip_next = True
                        merged_successfully = True

            if merged_successfully:
                continue

            # Step 4: Standard mapping for tokens that did not merge
            matched_in_dict = False

            if v_set and current_token in v_set:
                final_tokens.append(current_token)
                matched_in_dict = True
            elif v_list and len(current_token) >= 3:
                best_match = process.extractOne(current_token, v_list, scorer=fuzz.ratio)
                dynamic_threshold = 100.0 if len(current_token) <= 3 else 70.0

                if best_match and best_match[1] >= dynamic_threshold:
                    final_tokens.append(best_match[0])
                    matched_in_dict = True

            # Step 5: Fallback validation
            # Short words that didn't merge and aren't in the dictionary are permanently deleted.
            if not matched_in_dict:
                is_strong_token = len(current_token) >= 3
                is_nutritional_marker = current_token in nutritional_markers
                is_valid_nltk = self.english_vocab and current_token in self.english_vocab and len(current_token) >= 3

                if is_valid_nltk or is_strong_token or is_nutritional_marker:
                    final_tokens.append(current_token)

        # Destroy the entire chain if no token is long enough to be a real word anchor
        has_strong_anchor = False
        for token in final_tokens:
            if len(token) >= 4:
                has_strong_anchor = True
                break

        if not has_strong_anchor:
            return ""

        return " ".join(dict.fromkeys(final_tokens))

    def preprocess_image_domain(self, raw_image_rgb):
        """
        Abstract method. Must be implemented by child classes to handle
        domain-specific photometric issues.
        CRITICAL DUAL-STREAM: Must return (processed_image_for_yolo, raw_scaled_image_for_crops)
        """
        raise NotImplementedError("This method must be overridden in the specific strategy class.")

    def execute_detection(self, processed_image_rgb):
        """
        Abstract method. Must be implemented by child classes to perform
        the domain-specific detection (Multiscale for frontal, or Standard YOLO for cenital).
        """
        raise NotImplementedError("This method must be overridden in the specific strategy class.")

    def apply_early_spatial_deduplication(self, yolo_boxes):
        """
        Abstract method. Must be implemented by child classes (Frontal/Cenital)
        due to the differing physical occlusion dynamics of their perspectives.
        """
        raise NotImplementedError("This method must be overridden in the specific strategy class.")

    def apply_late_contextual_filtering(self, semantic_products):
        """
        Abstract method. Resolves physical nesting and stacking conflicts
        (e.g., printed logos vs stacked items) using semantic and OCR data.
        """
        raise NotImplementedError("This method must be overridden in the specific strategy class.")

    # --- MASTER EXECUTION PIPELINE ---

    def analyze_image(self, target_image_path):
        """
        Runs the full hybrid pipeline utilizing Dual-Stream architecture.
        Executes YOLO on the photometrically processed image to mitigate shadows/glares.
        Executes SAM and Multi-Orientation OCR on the raw unadulterated image to
        preserve high-frequency text textures.
        """
        import cv2
        import numpy as np

        print(f"Loading image: {target_image_path}")
        image_frame = cv2.imread(target_image_path)
        if image_frame is None:
            raise ValueError(f"Could not load image at {target_image_path}")

        image_frame_rgb = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)

        processed_image_rgb, raw_scaled_rgb = self.preprocess_image_domain(image_frame_rgb)

        phase_one_boxes = self.execute_detection(processed_image_rgb)

        if not phase_one_boxes:
            return raw_scaled_rgb, []

        phase_one_boxes = self.apply_early_spatial_deduplication(phase_one_boxes)

        extracted_candidates = []

        print("Stage 1.5: Extracting geometry and running rotational OCR...")
        for item in phase_one_boxes:
            box_coordinates = item['box']
            yolo_confidence = item['conf']

            x_min_initial, y_min_initial, width_initial, height_initial = box_coordinates
            absolute_box_initial = (x_min_initial, y_min_initial, x_min_initial + width_initial, y_min_initial + height_initial)

            # NOTE: box format changes here from XYWH (YOLO output) to XYXY (SAM output).
            # Everything downstream of this point uses XYXY coordinates.
            tight_box, pure_crop_image = self.refine_box_with_sam(raw_scaled_rgb, absolute_box_initial)
            base_crop_array = np.array(pure_crop_image)

            best_cleaned_text = ""
            best_raw_text = ""
            best_rotation_score = -1.0

            rotations = [
                (0, None),
                (90, cv2.ROTATE_90_CLOCKWISE),
                (180, cv2.ROTATE_180),
                (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ]

            for angle, cv2_rotation_flag in rotations:
                if angle == 0:
                    rotated_array = base_crop_array
                else:
                    rotated_array = cv2.rotate(base_crop_array, cv2_rotation_flag)

                crop_height = rotated_array.shape[0]
                # Only retain text blocks tall enough to be a product name or brand mark.
                # Fine print (nutritional facts, legal disclaimers) is typically < 8% of crop height.
                min_text_height = crop_height * 0.08

                # detail=1 returns (bbox, text, confidence) per detection, enabling
                # height-based filtering and confidence-weighted rotation scoring.
                raw_ocr_detections = self.ocr_reader.readtext(rotated_array, detail=1)

                large_text_detections = []
                for (bbox, text, conf) in raw_ocr_detections:
                    y_values = []
                    for point in bbox:
                        y_values.append(point[1])
                    text_height = max(y_values) - min(y_values)
                    if text_height >= min_text_height:
                        large_text_detections.append((bbox, text, conf))

                if not large_text_detections:
                    continue

                text_parts = []
                for (_, text, _) in large_text_detections:
                    text_parts.append(text)
                raw_text = " ".join(text_parts).lower()

                total_conf = 0.0
                for (_, _, conf) in large_text_detections:
                    total_conf += conf
                avg_confidence = total_conf / len(large_text_detections)

                cleaned_text = self.sanitize_and_correct_ocr_text(raw_text)
                valid_character_count = len(cleaned_text.replace(" ", ""))

                # Combined score weights both character richness and OCR confidence,
                # preventing a noisy rotation with many uncertain tokens from winning.
                rotation_score = valid_character_count * avg_confidence

                if rotation_score > best_rotation_score:
                    best_rotation_score = rotation_score
                    best_cleaned_text = cleaned_text
                    best_raw_text = raw_text

            extracted_candidates.append({
                'box': tight_box,
                'yolo_confidence': yolo_confidence,
                'crop': pure_crop_image,
                'extracted_text': best_cleaned_text,
                'raw_ocr': best_raw_text
            })

        print("Stage 2.0: Dense retrieval via DINOv2...")
        dense_products = self.semantic_engine.evaluate_candidates(
            extracted_candidates,
            maximum_distance_threshold=self.semantic_distance_threshold
        )

        print("Stage 2.1: Sparse retrieval and late fusion...")
        fused_products = self.fuse_hybrid_results(dense_products, extracted_candidates)

        print("Stage 2.5: OCR reranking...")
        fused_products = self.apply_local_ocr_reranking(fused_products)

        print("Stage 2.7: Late contextual filtering...")
        fused_products = self.apply_late_contextual_filtering(fused_products)

        print("Stage 3: Cross-validation scoring...")
        scored_products = self.calculate_cross_validation_scores(fused_products)

        print("Stage 4: Global deduplication...")
        final_products = self.apply_global_semantic_deduplication(scored_products, top_k_check=1)

        # Final audit log
        print("\n" + "="*95)
        print("FINAL RESULTS: Top-5 matches per detected product")
        print("="*95)
        for prod in final_products:
            print(f"\nBox: {prod['box']}  YOLO: {prod['yolo_confidence']:.2f}")
            if prod.get('raw_ocr'):
                print(f"  OCR: '{prod['raw_ocr']}' -> '{prod['extracted_text']}'")
            print(f"  {'Barcode':<15} | {'Source':<14} | {'Base':<6} | {'Bonus':<6} | {'Final':<6} | Product")
            print("  " + "-" * 80)
            for match in prod['top_matches'][:5]:
                barcode = match['barcode']
                name = self.product_text_database.get(barcode, "")[:35]
                source = match.get('source', 'DINOv2')
                base_conf = match.get('pre_rerank_conf', match['confidence_percentage'])
                bonus = match.get('add_bonus', 0.0)
                final_conf = match['confidence_percentage']
                print(f"  {barcode:<15} | {source:<14} | {base_conf:<6.1f} | +{bonus:<5.1f} | {final_conf:<6.1f} | {name}")
        print("="*95 + "\n")

        annotated_frame = cv2.cvtColor(raw_scaled_rgb, cv2.COLOR_RGB2BGR)
        for product in final_products:
            cx1, cy1, cx2, cy2 = product['box']
            best_match = product['top_matches'][0]
            barcode = best_match['barcode']
            confidence_percentage = best_match['confidence_percentage']

            cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
            label = f"{barcode} ({confidence_percentage:.1f}%)"
            cv2.putText(annotated_frame, label, (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated_frame, final_products
