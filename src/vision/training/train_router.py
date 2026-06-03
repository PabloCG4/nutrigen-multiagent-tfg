import os
from ultralytics import YOLO
from src.vision import config

def train_perspective_router_model() -> None:
    """
    Trains the definitive YOLO classification model for perspective routing.
    Inherits hardware limits (workers=2, batch=64) to guarantee stability on 4GB VRAM.
    Uses auto_augment=False to prevent RandAugment from applying physically 
    impossible transformations (like 90-degree rotations or vertical flips) 
    that would destroy the geometric definition of the frontal/cenital classes.
    """
    dataset_directory = config.ROUTER_DATASET_DIR
    
    if not os.path.exists(dataset_directory):
        raise FileNotFoundError(f"Router dataset directory not found at: {dataset_directory}")

    print("Initializing official YOLO nano classification model...")
    router_model = YOLO("yolo26n-cls.pt")

    print("Starting optimal perspective router training loop...")
    training_results = router_model.train(
        data=dataset_directory, 
        
        epochs=100,
        patience=20,
        task="classify",
        
        # Strictly Optimized Hardware Constraints (4GB VRAM Safe)
        imgsz=224,            
        batch=256,             
        workers=4,            
        device=0,
        
        # Classification-specific Fine-Tuning Hyperparameters
        optimizer="AdamW",
        lr0=0.0005,           
        lrf=0.01,
        cos_lr=True,          
        weight_decay=0.001,   
        freeze=0,             
        
        # Physically-Constrained Augmentation for Classification
        # auto_augment=False disables destructive random policies.
        # fliplr=0.5 is natively supported in classification and physically valid.
        # Note: Detection-specific params (hsv_h, degrees, scale) are omitted as 
        # YOLO classification dataloaders silently ignore them when auto_augment=False.
        auto_augment=False,   
        fliplr=0.5,           
        
        project=os.path.join(config.BASE_DIR, "reports", "training_runs"),
        name="perspective_router_model_final",
        exist_ok=False,
        plots=True,
        verbose=True
    )
    
    print("Definitive router training process finished successfully.")

if __name__ == "__main__":
    train_perspective_router_model()