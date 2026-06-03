import cv2
import numpy as np
from ultralytics import YOLO

from src.vision.common_pipeline import BaseSmartFridgeSystem

class CenitalSmartFridgeSystem(BaseSmartFridgeSystem):
    """
    Concrete Strategy for the Top-Down Perspective (Fridge Drawer).
    Implements localized shadow blending to prevent geometric fragmentation 
    across light/dark boundaries, and strict semantic absorption for nested elements.
    """
    
    def __init__(self, yolo_path, device=None):
        super().__init__(yolo_path, semantic_distance_threshold=0.45, device=device)
        print(f" Loading Cenital YOLO model from {self.yolo_path}...")
        self.detection_model_yolo = YOLO(self.yolo_path)

    def normalize_image_scale(self, image_array, target_height=1920):
        """
        Normalize the image scale to the target height.
        The target height is 1920 pixels.
        The image is resized using the LANCZOS4 interpolation.
        The image is returned as a numpy array.
        """
        original_height, original_width = image_array.shape[:2]
        if original_height == target_height:
            return image_array
            
        scaling_factor = target_height / float(original_height)
        new_width = int(original_width * scaling_factor)
        normalized_image = cv2.resize(image_array, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        return normalized_image

    def preprocess_image_domain(self, raw_image_rgb):
        """
        Restored to pure pass-through architecture.
        Photometric manipulation (CLAHE) caused severe Domain Shift, 
        destroying the Emmental detection and hallucinating boxes.
        """
        normalized_raw = self.normalize_image_scale(raw_image_rgb)
        return normalized_raw, normalized_raw

    def execute_detection(self, processed_image_rgb):
        """
        Executes a dual-resolution augmented inference pass (TTA at 640 and 1024).
        1024 resolution provides pixel-perfect bounding boxes for the OCR engine 
        and allows the FPN to bridge hard shadows (e.g., Delicias del Mar).
        640 resolution acts as a safety net for macro-objects.
        """
        print(" Evaluating image using dual-resolution native inference (TTA)...")
        
        # Base resolution pass (Macro-objects and standard features)
        base_inference = self.detection_model_yolo.predict(
            source=processed_image_rgb,
            conf=0.10,
            imgsz=640,
            augment=True,
            save=False,
            verbose=False
        )
        
        # High resolution pass (Micro-textures, shadow-bridging, and precise OCR boundaries)
        high_res_inference = self.detection_model_yolo.predict(
            source=processed_image_rgb,
            conf=0.10,
            imgsz=1024,
            augment=True,
            save=False,
            verbose=False
        )
        
        formatted_results = []
        
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
        Standard Non-Maximum Suppression (NMS). NMS is a algorithm to remove duplicate boxes. We use it because we are doing doble inference pass with different resolutions (640 and 1024).
        Threshold kept at 0.65 to prevent merging contiguous items.
        Leaves nested items intact for late semantic evaluation.
        """
        if not yolo_boxes:
            return []

        def get_confidence(box_dict):
            return box_dict['conf']
        
        # Sort the boxes by confidence in descending order 
        sorted_boxes = sorted(yolo_boxes, key=get_confidence, reverse=True)
        unique_boxes = []

        # Iterate through the sorted boxes, starting with the highest confidence box.
        for current_item in sorted_boxes:
            cx, cy, cw, ch = current_item['box']
            current_abs_box = (cx, cy, cx + cw, cy + ch)
            current_area = cw * ch
            is_duplicate = False

            # Calculate the IoU between the current box and the boxes in the unique_boxes list (accepted boxes). Doing so, we keep the box with the highest confidence, and remove the duplicates.
            for kept_item in unique_boxes:
                kx, ky, kw, kh = kept_item['box']
                kept_abs_box = (kx, ky, kx + kw, ky + kh)
                
                x_min_inter = max(current_abs_box[0], kept_abs_box[0])
                y_min_inter = max(current_abs_box[1], kept_abs_box[1])
                x_max_inter = min(current_abs_box[2], kept_abs_box[2])
                y_max_inter = min(current_abs_box[3], kept_abs_box[3])
                # Intersection area is the area of the intersection of the two boxes.
                inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
                # Union area is the area of the union of the two boxes.
                union_area = current_area + (kw * kh) - inter_area
                
                # IoU is the intersection over the union of the two boxes. 
                if union_area > 0 and (inter_area / union_area) > 0.65:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_boxes.append(current_item)

        print(f" Stage 1.2: Early NMS filtering: {len(yolo_boxes)} -> {len(unique_boxes)} items.")
        return unique_boxes

    def apply_late_contextual_filtering(self, semantic_products):
        """
        Pure pass-through architecture for the Cenital domain.
        Since top-down perspectives naturally involve smaller items (e.g., kiwis) 
        physically stacked on top of larger items (e.g., meat trays), utilizing 
        Intersection over Minimum (IoM) logic causes fatal false positives 
        (deleting the stacked item due to OCR bounding box cross-pollution).
        
        The new fine-tuned YOLO model natively resolves fragmented boxes 
        (like Mozzarella bags), making heuristic contextual deletion obsolete 
        and dangerous for this specific camera angle.
        """
        return semantic_products

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from src.vision import config

    TEST_IMAGE_PATH = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\data\vision\test_images\image_a90019.jpg"

    try:
        print(f"\n Starting isolated Cenital strategy test...")
        vision_system = CenitalSmartFridgeSystem(yolo_path=config.CENITAL_MODEL_V26_FINETUNED)
        
        result_img, products = vision_system.analyze_image(TEST_IMAGE_PATH)

        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(16, 12))
        plt.imshow(result_img_rgb)
        plt.axis('off')
        plt.title("Cenital Strategy - Dual Stream Pipeline")
        plt.tight_layout()
        plt.show()

    except Exception as error:
        print(f"error: {error}")