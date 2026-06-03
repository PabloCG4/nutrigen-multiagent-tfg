import os
import shutil
import sys
from pathlib import Path
from src.vision import config

def build_safe_mvtec_dataset(raw_images_dir: str, yolo_labels_dir: str, output_dataset_dir: str):
    """
    Constructs the final MVTec dataset using a zero-risk copy approach.
    It reads raw images and ground truth labels, validates matches in memory,
    and copies valid pairs to the output directory. Source files are never modified or deleted.
    """
    raw_images_path = Path(raw_images_dir)
    labels_base_path = Path(yolo_labels_dir)
    output_base_path = Path(output_dataset_dir)

    if not raw_images_path.exists():
        print(f"Critical error: Raw images directory not found at {raw_images_path}")
        sys.exit(1)

    labels_train_path = labels_base_path / "train"
    labels_val_path = labels_base_path / "val"

    if not labels_train_path.exists() or not labels_val_path.exists():
        print("Critical error: Ground truth label directories not found.")
        sys.exit(1)

    print("Cataloging raw images into memory...")
    raw_image_catalog = {}
    for filename in os.listdir(raw_images_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_stem = Path(filename).stem
            raw_image_catalog[file_stem] = raw_images_path / filename

    total_raw_images = len(raw_image_catalog)
    print(f"Total raw images cataloged: {total_raw_images}")

    if total_raw_images == 0:
        print("Critical error: No images found in the raw directory. Aborting.")
        sys.exit(1)

    target_images_train = output_base_path / "images" / "train"
    target_images_val = output_base_path / "images" / "val"
    target_labels_train = output_base_path / "labels" / "train"
    target_labels_val = output_base_path / "labels" / "val"

    def process_dataset_split(split_name: str, source_labels: Path, target_images: Path, target_labels: Path):
        print(f"\nEvaluating {split_name.upper()} split...")
        
        expected_labels_list = [file_name for file_name in os.listdir(source_labels) if file_name.endswith('.txt')]
        if len(expected_labels_list) == 0:
            print(f"Critical error: No labels found in {source_labels}. Aborting.")
            sys.exit(1)

        matched_pairs_count = 0
        missing_images_count = 0

        for label_filename in expected_labels_list:
            label_stem = Path(label_filename).stem
            if label_stem in raw_image_catalog:
                matched_pairs_count += 1
            else:
                missing_images_count += 1

        print(f"Pre-check results for {split_name}: {matched_pairs_count} valid matches, {missing_images_count} missing images.")

        if matched_pairs_count == 0:
            print("CRITICAL ERROR: Zero image-label matches found.")
            print("Aborting execution to prevent invalid dataset generation.")
            sys.exit(1)

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        copied_files_count = 0
        for label_filename in expected_labels_list:
            label_stem = Path(label_filename).stem
            if label_stem in raw_image_catalog:
                source_image_file = raw_image_catalog[label_stem]
                source_label_file = source_labels / label_filename
                
                destination_image_file = target_images / source_image_file.name
                destination_label_file = target_labels / label_filename
                
                shutil.copy2(source_image_file, destination_image_file)
                shutil.copy2(source_label_file, destination_label_file)
                copied_files_count += 1

        print(f"Successfully constructed {split_name.upper()} split with {copied_files_count} paired files.")

    process_dataset_split("train", labels_train_path, target_images_train, target_labels_train)
    process_dataset_split("val", labels_val_path, target_images_val, target_labels_val)

    print("\nDataset construction completed successfully. Source files remain intact.")

if __name__ == "__main__":
    RAW_IMAGES_DIRECTORY = os.path.join(config.VISION_DATA_DIR, "mvtec_raw", "images")
    YOLO_LABELS_DIRECTORY = os.path.join(config.VISION_DATA_DIR, "mvtec_raw", "yolo_labels")
    FINAL_DATASET_DIRECTORY = os.path.join(config.VISION_DATA_DIR, "mvtec_dataset")
    
    build_safe_mvtec_dataset(RAW_IMAGES_DIRECTORY, YOLO_LABELS_DIRECTORY, FINAL_DATASET_DIRECTORY)