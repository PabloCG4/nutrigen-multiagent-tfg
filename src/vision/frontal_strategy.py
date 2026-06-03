import cv2
import numpy as np
from ultralytics import YOLO

from src.vision.common_pipeline import BaseSmartFridgeSystem

class FrontalSmartFridgeSystem(BaseSmartFridgeSystem):
    """
    Concrete Strategy for the Frontal Perspective (Pantry/Fridge).
    Stripped of legacy over-engineering (SAHI/WBF). Relies on the native 
    multiscale feature extraction of the converged YOLO26M model.
    """

    def __init__(self, yolo_path, device=None):
        super().__init__(yolo_path, semantic_distance_threshold=0.65, device=device)
        print(f" Loading Frontal YOLO model from {self.yolo_path}...")
        self.detection_model_yolo = YOLO(self.yolo_path)

    def normalize_image_scale(self, image_array, target_height=1920):
        original_height, original_width = image_array.shape[:2]
        if original_height == target_height:
            return image_array
            
        scaling_factor = target_height / float(original_height)
        new_width = int(original_width * scaling_factor)
        normalized_image = cv2.resize(image_array, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        return normalized_image

    def preprocess_image_domain(self, raw_image_rgb):
        """
        Pure pass-through architecture. 
        The converged model is natively robust to LED lighting and glare.
        """
        normalized_image = self.normalize_image_scale(raw_image_rgb)
        return normalized_image, normalized_image

    def execute_detection(self, processed_image_rgb):
        """
        Executes a dual-resolution native inference pass to capture both 
        macro-objects and micro-textures without relying on aggressive 
        image tiling. Results are concatenated for subsequent spatial filtering.
        """
        print(" Evaluating image using dual-resolution native inference...")
        
        # Base resolution pass (captures standard and large items)
        base_inference = self.detection_model_yolo.predict(
            source=processed_image_rgb,
            conf=0.15,
            imgsz=640,
            save=False,
            verbose=False
        )
        
        # High resolution pass (captures micro-textures and dense zones)
        high_res_inference = self.detection_model_yolo.predict(
            source=processed_image_rgb,
            conf=0.15,
            imgsz=1024,
            save=False,
            verbose=False
        )
        
        formatted_results = []
        
        # Extract and format boxes from both inference passes
        for inference_batch in [base_inference, high_res_inference]:
            detected_boxes_yolo = inference_batch[0].boxes
            for box_data in detected_boxes_yolo:
                coordinates = box_data.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, coordinates)
                confidence = float(box_data.conf[0].cpu().numpy())
                box_width = x_max - x_min
                box_height = y_max - y_min
                
                formatted_results.append({
                    'box': [x_min, y_min, box_width, box_height],
                    'conf': confidence
                })
                
        return formatted_results

    def apply_early_spatial_deduplication(self, yolo_boxes):
        """
        Standard Intersection over Union (IoU) Non-Maximum Suppression (NMS)
        to resolve duplicate predictions originating from the dual inference pass.

        Note: The IoM-based Fragment Suppression stage (Stage 1) has been moved
        to apply_late_contextual_filtering, where barcode information is available.
        Suppressing nested boxes here (before semantic evaluation) is unsafe because
        a small box nested inside a larger one may be a genuinely different product
        (e.g., a yogurt pot inside a tray). Deferring the decision until barcodes
        are known prevents false deletions.
        """
        if not yolo_boxes:
            return []

        # --- Stage 1: IoM Fragment Suppression (DISABLED) ---
        # Moved to apply_late_contextual_filtering where barcode data is available.
        # Deleting a nested box here (before semantic evaluation) risks removing a
        # real product that happens to sit physically inside another YOLO box.
        # The late filter performs the same IoM check but only deletes if both boxes
        # also share barcodes, making the decision semantically grounded.
        #
        # def calculate_area(box_dictionary):
        #     return box_dictionary['box'][2] * box_dictionary['box'][3]
        #
        # boxes_sorted_by_area = sorted(yolo_boxes, key=calculate_area, reverse=True)
        # container_filtered_boxes = []
        #
        # for current_item in boxes_sorted_by_area:
        #     cx, cy, cw, ch = current_item['box']
        #     current_area = float(cw * ch)
        #     is_fragment = False
        #
        #     for kept_item in container_filtered_boxes:
        #         kx, ky, kw, kh = kept_item['box']
        #         x_min_intersection = max(cx, kx)
        #         y_min_intersection = max(cy, ky)
        #         x_max_intersection = min(cx + cw, kx + kw)
        #         y_max_intersection = min(cy + ch, ky + kh)
        #         intersection_width = max(0.0, float(x_max_intersection - x_min_intersection))
        #         intersection_height = max(0.0, float(y_max_intersection - y_min_intersection))
        #         intersection_area = intersection_width * intersection_height
        #         if current_area > 0 and (intersection_area / current_area) > 0.70:
        #             is_fragment = True
        #             break
        #     if not is_fragment:
        #         container_filtered_boxes.append(current_item)

        # Stage 2: Standard NMS (IoU) for overlapping duplicates from the dual inference pass
        boxes_sorted_by_confidence = sorted(yolo_boxes, key=lambda dictionary: dictionary['conf'], reverse=True)
        final_unique_boxes = []

        for current_item in boxes_sorted_by_confidence:
            cx, cy, cw, ch = current_item['box']
            current_area = float(cw * ch)
            is_duplicate = False

            for kept_item in final_unique_boxes:
                kx, ky, kw, kh = kept_item['box']
                kept_area = float(kw * kh)
                
                x_min_intersection = max(cx, kx)
                y_min_intersection = max(cy, ky)
                x_max_intersection = min(cx + cw, kx + kw)
                y_max_intersection = min(cy + ch, ky + kh)

                intersection_width = max(0.0, float(x_max_intersection - x_min_intersection))
                intersection_height = max(0.0, float(y_max_intersection - y_min_intersection))
                intersection_area = intersection_width * intersection_height
                
                union_area = current_area + kept_area - intersection_area
                intersection_over_union = intersection_area / union_area if union_area > 0 else 0.0
                
                # Standard threshold to merge highly overlapping predictions
                if intersection_over_union > 0.65:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_unique_boxes.append(current_item)

        print(f" Stage 1.2: IoU NMS deduplication: {len(yolo_boxes)} -> {len(final_unique_boxes)} items.")
        return final_unique_boxes

    def apply_late_contextual_filtering(self, semantic_products):
        """
        Solves occlusion dynamically via DINOv2 confidence check.
        If IoM > 80% and they share Top-50 barcodes, the weaker box is destroyed.
        Better than IoM from early spatial deduplication because it uses the confidence of the top match and the barcodes of the top matches.
        """
        # semantic_products is a list of products with the following structure:
        # {
        #     'box': [x_min, y_min, x_max, y_max],
        #     'conf': confidence,
        #     'top_matches': [
        #         {
        #             'barcode': barcode,
        #             'confidence_percentage': confidence_percentage
        #         }
        #     ]
        # }
        # The products are from the semantic engine.
        if not semantic_products:
            return []

        indices_to_remove = set()

        for i in range(len(semantic_products)):
            if i in indices_to_remove:
                continue

            prod_i = semantic_products[i]
            box_i = prod_i['box']
            conf_i = prod_i['top_matches'][0].get('pre_rerank_conf', prod_i['top_matches'][0]['confidence_percentage'])

            barcodes_i = set()
            for match in prod_i['top_matches']:
                barcodes_i.add(match['barcode'])

            for j in range(i + 1, len(semantic_products)):
                if j in indices_to_remove:
                    continue

                prod_j = semantic_products[j]
                box_j = prod_j['box']
                conf_j = prod_j['top_matches'][0].get('pre_rerank_conf', prod_j['top_matches'][0]['confidence_percentage'])

                barcodes_j = set()
                for match in prod_j['top_matches']:
                    barcodes_j.add(match['barcode'])

                iom = self.calculate_intersection_over_minimum(box_i, box_j)

                if iom >= 0.80 and not barcodes_i.isdisjoint(barcodes_j):
                    if conf_i < conf_j:
                        indices_to_remove.add(i)
                        break
                    else:
                        indices_to_remove.add(j)

        valid_products = []
        for idx in range(len(semantic_products)):
            if idx not in indices_to_remove:
                valid_products.append(semantic_products[idx])

        print(f" Stage 2.7: Contextual filtering removed {len(indices_to_remove)} overlapping boxes.")
        return valid_products