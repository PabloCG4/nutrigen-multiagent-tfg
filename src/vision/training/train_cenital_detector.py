import os
from ultralytics import YOLO
from src.vision import config

def train_robust_agnostic_model():
    """
    Trains a YOLO model for agnostic product detection (class 0).
    Utilizes a hybrid dataset (MVTec D2S + RPC) under top-down constraints.
    Hyperparameters are strictly optimized for real-world kitchen environments,
    accounting for imperfect camera angles, countertop reflections, and 
    inconsistent lighting conditions.
    """
    yaml_configuration_path = os.path.join(config.VISION_DATA_DIR, "mvtec_dataset", "dataset.yaml")
    
    base_weights_path = "yolo26m.pt"

    print(f"Initializing base YOLO model from {base_weights_path}...")
    vision_model = YOLO(base_weights_path) 

    print("Starting robust top-down training loop...")
    training_results = vision_model.train(
        data=yaml_configuration_path,
        
        epochs=100,            
        patience=50,          
        
        # Hardware Constraints (Optimized for 4GB VRAM)
        imgsz=640,            
        batch=3,              
        workers=2,            
        device=0,             
        
        # Real-World Environment Augmentation Strategy
        scale=0.2,            # Slight scale variation for different countertop distances
        perspective=0.002,    # Subtle perspective distortion for imperfect user camera angles
        mosaic=0.5,           # Enforces object feature extraction over background memorization
        mixup=0.0,            # Disabled: physical objects remain opaque
        degrees=180.0,        # Kept: objects can be placed in any rotation on the 2D plane
        fliplr=0.5,           
        flipud=0.5,           
        
        # Refined lighting augmentation for subtle kitchen variations
        hsv_h=0.015,          # Subtle hue variance
        hsv_s=0.5,            # Reduced saturation variance for realistic reflections
        hsv_v=0.4,            # Reduced value variance for realistic shadows
        
        label_smoothing=0.1,  
        
        project=os.path.join(config.BASE_DIR, "reports", "training_runs"),
        name="agnostic_detector_cenital", 
        exist_ok=False,       
        plots=True,
        verbose=True
    )

    print("Robust training process finished successfully.")

if __name__ == "__main__":
    train_robust_agnostic_model()