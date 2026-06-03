import os
import sys
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time

from src.vision import config
from src.vision.frontal_strategy import FrontalSmartFridgeSystem
from src.vision.cenital_strategy import CenitalSmartFridgeSystem

# Shared router weights path — used by both the display and programmatic functions
ROUTER_WEIGHTS = os.path.join(
    config.BASE_DIR,
    "reports", "training_runs",
    "perspective_router_model_final2", "weights", "best.pt",
)


def extract_products_from_image(image_path: str) -> list:
    """
    Runs the full vision pipeline on a single image
    and returns the raw list of detected product dicts produced by analyze_image().
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(ROUTER_WEIGHTS):
        raise RuntimeError(
            f"Perspective router weights not found at: {ROUTER_WEIGHTS}"
        )

    perspective_router = YOLO(ROUTER_WEIGHTS)

    print(f" Analysing perspective for: {image_path}")
    router_predictions = perspective_router(image_path, verbose=False)
    top_class_index = router_predictions[0].probs.top1
    detected_perspective = router_predictions[0].names[top_class_index]
    print(f" Routing decision: {detected_perspective.upper()}")

    # Release router from memory before loading the heavy strategy model
    del perspective_router

    if detected_perspective == "frontal":
        vision_system = FrontalSmartFridgeSystem(yolo_path=config.FRONTAL_MODEL_V26)
    elif detected_perspective == "cenital":
        vision_system = CenitalSmartFridgeSystem(yolo_path=config.CENITAL_MODEL_V26_FINETUNED)
    else:
        raise RuntimeError(
            f"Unknown perspective '{detected_perspective}' returned by router."
        )

    _annotated_frame, final_products = vision_system.analyze_image(image_path)
    print(f"[Vision] Pipeline complete. {len(final_products)} product(s) validated.")
    return final_products


def run_single_image_inference(target_image_path: str) -> None:
    """
    Master inference pipeline for a single static image.
    Uses a nano-classifier to determine the physical perspective, and dynamically 
    instantiates ONLY the necessary semantic strategy to preserve VRAM limits.
    """
    total_start = time.perf_counter()

    if not os.path.exists(target_image_path):
        print(f"Critical Error: Input image not found at {target_image_path}")
        sys.exit(1)

    if not os.path.exists(ROUTER_WEIGHTS):
        print(f"Critical Error: Router weights not found at {ROUTER_WEIGHTS}")
        sys.exit(1)

    print(" Allocating Nano Perspective Router in VRAM...")
    perspective_router = YOLO(ROUTER_WEIGHTS)

    print(f" Analyzing geometric perspective for: {target_image_path}")
    router_predictions = perspective_router(target_image_path, verbose=False)
    
    top_class_index = router_predictions[0].probs.top1
    detected_perspective = router_predictions[0].names[top_class_index]
    
    print(f" Routing Decision: {detected_perspective.upper()} detected.")

    # Dynamic Instantiation: Only the required heavy models are loaded into VRAM
    if detected_perspective == "frontal":
        print(" Instantiating Frontal Strategy System. This may take a few seconds...")
        vision_system = FrontalSmartFridgeSystem(yolo_path=config.FRONTAL_MODEL_V26)
        
    elif detected_perspective == "cenital":
        print(" Instantiating Cenital Strategy System. This may take a few seconds...")
        vision_system = CenitalSmartFridgeSystem(yolo_path=config.CENITAL_MODEL_V26_FINETUNED)
        
    else:
        print(f" Unknown perspective '{detected_perspective}' predicted by router.")
        sys.exit(1)

    annotated_image_bgr, final_products = vision_system.analyze_image(target_image_path)
    print(f" Processing complete. Validated {len(final_products)} products.")
    
    # Increase the text size for high resolutions
    cv2.putText(annotated_image_bgr, f"Perspective: {detected_perspective.upper()}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    
    # Matplotlib requires RGB instead of BGR
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(16, 12))
    plt.imshow(annotated_image_rgb)
    plt.title(f"Master Kitchen Assistant - Static Inference ({detected_perspective.upper()})", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    total_elapsed = time.perf_counter() - total_start
    print(f" Vision total execution time: {total_elapsed:.2f}s")
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\data\vision\test_images\fridge_box.jpg"
    
    run_single_image_inference(TEST_IMAGE)