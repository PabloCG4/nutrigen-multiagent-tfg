import os
from pathlib import Path

# --- 1. PROJECT ROOT & PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- 2. DATA DIRECTORIES ---
DATA_DIR = os.path.join(BASE_DIR, "data")
VISION_DATA_DIR = os.path.join(DATA_DIR, "vision")

# --- 3. SHARED VISION DIRECTORIES ---

# A. Open Food Facts (OFF) - Fine-Grained Classification
# Stores images scraped from OFF (Product ID recognition)
IMAGES_DIR = os.path.join(VISION_DATA_DIR, "off_images")

# B. OpenImages V7 - Object Detection
# Stores raw data downloaded via FiftyOne (Stage 1 training data)
OPENIMAGES_RAW_DIR = os.path.join(VISION_DATA_DIR, "openimages_raw")

# Stores metadata files (Hierarchy JSON & Class Descriptions CSV)
OPENIMAGES_META_DIR = os.path.join(VISION_DATA_DIR, "metadata")

# C. YOLO Dataset - Processed Data
# Stores the final dataset formatted for YOLOv8 training (images/labels structure)
YOLO_DATASET_DIR = os.path.join(VISION_DATA_DIR, "yolo_dataset")
YOLO_MODEL_DIR = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\data\vision\yolo_dataset\yolov8n.pt" 
YOLO_MODEL_V26 = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\reports\training_runs\agnostic_detector_medium_v12_final\weights\last.pt"
CENITAL_MODEL_V26 = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\reports\training_runs\agnostic_detector_cenital3\weights\last.pt"
CENITAL_MODEL_V26_FINETUNED = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\reports\training_runs\agnostic_detector_cenital_finetuned_12032\weights\last.pt"
FRONTAL_MODEL_V26 = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\reports\training_runs\agnostic_detector_frontal_final\weights\last.pt"
ROUTER_DATASET_DIR = os.path.join(VISION_DATA_DIR, "router_dataset")
FRONTAL_DATASET_DIR = os.path.join(VISION_DATA_DIR, "dataset_frontal_unificado")

# --- 5. TRAINING ARTIFACTS ---
CLASSIFICATION_DATASET_DIR = os.path.join(DATA_DIR, "vision", "classification_dataset_resized")
CLASSIFICATION_DATASET_DIR_ORIGINAL = os.path.join(DATA_DIR, "vision", "classification_dataset")
DETECTION_DATASET_RAW_DIR = os.path.join(DATA_DIR, "vision", "raw_datasets_detection")

EMBEDDINGS_MODEL_DIR = os.path.join(DATA_DIR, "models", "embeddings")
CLASSIFIER_MODEL_DIR = os.path.join(DATA_DIR, "models", "classifier")

SAM_MODEL_DIR = os.path.join(DATA_DIR, "models", "segmentation")

OUTPUT_MVTEC_TXT_DIRECTORY = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\data\vision\mvtec_raw\yolo_labels\train"
IMAGES_MVTEC_DIRECTORY = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\data\vision\mvtec_raw\images"

