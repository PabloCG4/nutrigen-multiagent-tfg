import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.vision import config

INPUT_DIR = Path(config.CLASSIFICATION_DATASET_DIR_ORIGINAL)
OUTPUT_DIR = INPUT_DIR.parent / "classification_dataset_resized"
TARGET_SIZE = (256, 256) # Slightly larger than 224 for augmentation cropping

def process_image(file_info):
    """
    Resizes a single image and saves it to the new location.
    """
    source_path = file_info[0]
    destination_path = file_info[1]
    
    try:
        original_image = Image.open(source_path)
        # Convert to RGB (handle PNGs with transparency or Grayscale)
        processed_image = original_image.convert('RGB')
        # High-quality resize
        processed_image = processed_image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        # Save as JPEG with optimized size
        processed_image.save(destination_path, "JPEG", quality=85, optimize=True)
        original_image.close()
        processed_image.close()
            
    except Exception as exception_error:
        print(f"Error processing {source_path.name}: {exception_error}")

def run_resize():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    print(f"Resizing images from {INPUT_DIR} to {OUTPUT_DIR}...")
    
    # Collect all image files
    tasks = []
    
    # Walk through the dataset structure (barcode_folders/images)
    # We use list() to get all files first
    class_folders = []
    for folder_path in INPUT_DIR.iterdir():
        if folder_path.is_dir():
            class_folders.append(folder_path)
    
    for folder in class_folders:
        # Create corresponding folder in output
        destination_folder = OUTPUT_DIR / folder.name
        destination_folder.mkdir(parents=True, exist_ok=True)
        
        # Iterate clearly through all files
        for image_file in folder.glob("*.*"):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            if image_file.suffix.lower() in valid_extensions:
                source_file = image_file
                new_file_name = image_file.with_suffix(".jpg").name
                destination_file = destination_folder / new_file_name
                
                # Add task tuple to the list
                tasks.append((source_file, destination_file))

    print(f"Found {len(tasks)} images. Starting parallel processing...")
    
    # Use ThreadPool to maximize I/O speed
    thread_executor = ThreadPoolExecutor(max_workers=8)
    
    # Submit tasks to the executor clearly
    pending_tasks = []
    for task_arguments in tasks:
        task_future = thread_executor.submit(process_image, task_arguments)
        pending_tasks.append(task_future)
        
    # Manage progress bar and execution
    progress_bar = tqdm(pending_tasks, total=len(tasks), unit="img")
    for completed_task in progress_bar:
        completed_task.result()
        
    # Safely close the thread pool
    thread_executor.shutdown(wait=True)

    print("\n Dataset resized successfully")
    print(f"New location: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_resize()