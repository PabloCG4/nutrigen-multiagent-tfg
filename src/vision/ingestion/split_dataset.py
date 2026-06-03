import random
import shutil
from pathlib import Path
from src.vision import config

def split_data_for_training(training_ratio: float = 0.85):
    """
    Performs a stratified split of the dataset to ensure images from OpenImages, 
    SKU-110K, and Fridge Objects are represented in both train and val sets.
    """
    base_dataset_path = Path(config.YOLO_DATASET_DIR)
    
    # Define source and destination paths for images and labels
    validation_images_directory = base_dataset_path / "images" / "val"
    validation_labels_directory = base_dataset_path / "labels" / "val"
    
    training_images_directory = base_dataset_path / "images" / "train"
    training_labels_directory = base_dataset_path / "labels" / "train"

    # Create destination directories
    training_images_directory.mkdir(parents=True, exist_ok=True)
    training_labels_directory.mkdir(parents=True, exist_ok=True)

    # Get all image files currently in the val folder
    all_validation_images = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for file_path in validation_images_directory.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            all_validation_images.append(file_path)
    
    # Categorize images by source using prefixes established during ingestion
    sku_dataset_images = []
    fridge_dataset_images = []
    openimages_dataset_images = []
    
    for image_file in all_validation_images:
        file_name = image_file.name
        if file_name.startswith("sku_"):
            sku_dataset_images.append(image_file)
        elif file_name.startswith("fridge_"):
            fridge_dataset_images.append(image_file)
        else:
            openimages_dataset_images.append(image_file)

    dataset_categories = {
        "SKU-110K": sku_dataset_images,
        "Fridge Objects": fridge_dataset_images,
        "OpenImages": openimages_dataset_images
    }

    print(f"Starting stratified split with training ratio: {training_ratio}")

    for category_name, category_image_list in dataset_categories.items():
        if len(category_image_list) == 0:
            print(f"No images found for category: {category_name}")
            continue

        # Calculate number of images to move to training for this specific category
        # This maintains the stratified distribution across sources
        total_category_images = len(category_image_list)
        number_of_images_to_move = int(total_category_images * training_ratio)
        
        # Select a random sample of images to transfer to the training set
        images_selected_for_training = random.sample(category_image_list, number_of_images_to_move)

        print(f"Moving {number_of_images_to_move}/{total_category_images} images from {category_name} to train.")

        for selected_image_path in images_selected_for_training:
            # Move image file to the training directory
            destination_image_path = training_images_directory / selected_image_path.name
            shutil.move(str(selected_image_path), str(destination_image_path))
            
            # Move corresponding label file if it exists
            label_file_name = f"{selected_image_path.stem}.txt"
            source_label_path = validation_labels_directory / label_file_name
            destination_label_path = training_labels_directory / label_file_name
            
            if source_label_path.exists():
                shutil.move(str(source_label_path), str(destination_label_path))

    final_training_count = 0
    for item in training_images_directory.iterdir():
        final_training_count += 1 
    final_validation_count = 0
    for item in validation_images_directory.iterdir():
        final_validation_count += 1 
    print(f"Total images in TRAIN: {final_training_count}")
    print(f"Total images in VAL: {final_validation_count}")

if __name__ == "__main__":
    split_data_for_training(training_ratio=0.85)