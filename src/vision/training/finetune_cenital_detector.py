import os
from ultralytics import YOLO
from src.vision import config

def train_definitive_cenital_model():
    """
    Executes the definitive fine-tuning pipeline for the top-down (cenital) detector.
    Resolves Catastrophic Forgetting and the Shadow-Split problem by utilizing
    Random Erasing and deep mosaic composition, while safeguarding VRAM limits.
    """
    yaml_configuration_path = os.path.join(config.VISION_DATA_DIR, "cenital_finetuning_manualimages", "data.yaml")
    
    # Load the model that needs fine-tuning
    base_weights_path = config.CENITAL_MODEL_V26

    print(f"[INFO] Initializing base YOLO model from {base_weights_path}...")
    vision_model = YOLO(base_weights_path) 

    print("[INFO] Starting definitive top-down training loop...")
    training_results = vision_model.train(
        data=yaml_configuration_path,
        
        # --- Training Duration & Stability ---
        # Increased to 100 to allow the Auto-optimizer to execute a full OneCycleLR schedule
        epochs=100,           
        patience=25,                 
        
        # --- Hardware Constraints (Strict 4GB VRAM profile) ---
        imgsz=640,            
        batch=3,              # Reduced to absolutely guarantee no OOM during complex augmentations
        workers=2,            
        device=0,             
        
        # --- Deep Adaptation ---
        freeze=0,             # Unfrozen to allow base filters to adapt to LED lighting and plastic glare
        # Optimizer and LR parameters are intentionally omitted to allow the engine 
        # to dynamically calculate the optimal learning rate curve.

        # --- Anti-Shadow & Anti-Overfitting Augmentations ---
        erasing=0.4,          # CRITICAL: Forces network to infer full objects even when cut by hard shadows
        mosaic=1.0,           # Destroys rigid background context to prevent environment overfitting
        close_mosaic=15,      # Restores real-world continuous geometry for the final fine-tuning phase
        
        # --- Spatial & Color Augmentations ---
        scale=0.3,            # Increased slightly to handle zooming differences between drawers and countertops
        perspective=0.001,    # Handles imperfect smartphone capture angles
        degrees=15.0,         # Restricts rotation to avoid completely inverting shadows
        fliplr=0.5,           
        flipud=0.0,           # Disabled to preserve the directional logic of fridge top-down LED lighting
        mixup=0.0,            # Disabled to ensure physical products remain solid and opaque
        
        hsv_h=0.015,          
        hsv_s=0.5,            
        hsv_v=0.4,            
        
        # --- Output Logging ---
        project=os.path.join(config.BASE_DIR, "reports", "training_runs"),
        name="agnostic_detector_cenital_finetuned_1203", 
        exist_ok=False,       
        plots=True,
        verbose=True
    )

    print("[INFO] Definitive training process finished successfully.")

if __name__ == "__main__":
    train_definitive_cenital_model()