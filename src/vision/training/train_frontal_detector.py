import os
from ultralytics import YOLO
from src.vision import config

def run_agnostic_frontal_training():
    """
    Executes the definitive training pipeline for the agnostic frontal detector.
    Optimized for a balanced dataset (fridges, pantries, shelves, and SKU images).
    Prioritizes high-visibility bounding boxes to guarantee downstream semantic 
    classification via DINOv2 and OCR.
    """
    # Assuming the YAML file is placed where the configuration points
    yaml_path = os.path.join(config.FRONTAL_DATASET_DIR, "dataset.yaml")
    
    # Base weights for Medium model
    weights_path = "yolo26m.pt"

    print("[INFO] Initializing YOLO Medium model for agnostic detection...")
    vision_model = YOLO(weights_path) 

    print("[INFO] Starting definitive training loop...")
    results = vision_model.train(
        data=yaml_path,
        
        # --- Training Duration & Checkpointing ---
        epochs=100,
        patience=15,
        save_period=10,           # Saves a checkpoint model every 10 epochs
        
        # --- Hardware Configuration ---
        imgsz=640,
        batch=3,
        workers=2,
        device=0,
        
        # --- Architecture Initialization ---
        pretrained=True,
        freeze=0,
        
        # --- Spatial & Color Augmentation ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,             # Reduced rotation to maintain text orientation
        perspective=0.0005,       # Critical for angled pantry shelves
        scale=0.5,
        fliplr=0.5,
        
        # --- Composition Augmentation ---
        mosaic=1.0,
        close_mosaic=15,          # Disables mosaic for the final 15 epochs to stabilize real-world geometry
        copy_paste=0.0,           # Disabled: Downstream OCR/DINOv2 requires non-occluded products
        mixup=0.05,               # Minimal blend strictly for background robustness
        overlap_mask=True,
        
        # --- Regularization ---
        label_smoothing=0.05,     # Lowered for single-class (agnostic) confidence boosting
        
        # --- Output Logging ---
        project=os.path.join(config.BASE_DIR, "reports", "training_runs"),
        name="agnostic_detector_frontal_final",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    print("[INFO] Training process finished successfully.")

#if __name__ == "__main__":
 #   run_agnostic_frontal_training()


def resume_agnostic_frontal_training():
    """
    Resumes the interrupted training from the last saved checkpoint.
    Ultralytics will automatically load the exact epoch, learning rate, 
    and mosaic scheduling from the last.pt weights.
    """
    # Route to the interrupted checkpoint
    interrupted_weights_path = os.path.join(
        config.BASE_DIR, "reports", "training_runs", "agnostic_detector_frontal_final", "weights", "last.pt"
    )

    if not os.path.exists(interrupted_weights_path):
        raise FileNotFoundError(f"[CRITICAL ERROR] Checkpoint not found at {interrupted_weights_path}")

    print(f"[INFO] Resuming training from checkpoint: {interrupted_weights_path}...")
    
    # Load the partially trained model
    vision_model = YOLO(interrupted_weights_path) 

    # Resume the process. 'resume=True' explicitly tells the engine to ignore new parameters 
    # and continue exactly where it left off, ensuring close_mosaic triggers correctly.
    results = vision_model.train(resume=True)

    print("[INFO] Resumed training process finished successfully.")

if __name__ == "__main__":
    resume_agnostic_frontal_training()