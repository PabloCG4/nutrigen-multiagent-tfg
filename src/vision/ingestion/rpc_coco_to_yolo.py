import os
import json
import shutil
import random
from pathlib import Path
from src.vision import config


def convert_and_sample_rpc_with_prefix(
    json_path: str,
    images_source_dir: str,
    target_base_dir: str,
    split_name: str,
    max_samples: int
):
    """
    Converts RPC COCO annotations to YOLO format (class 0) and injects them 
    into the unified dataset. Uses an 'rpc_' prefix to prevent filename 
    collisions with MVTec data, ensuring total dataset integrity.
    """
    target_images_path = Path(target_base_dir) / "images" / split_name
    target_labels_path = Path(target_base_dir) / "labels" / split_name

    target_images_path.mkdir(parents=True, exist_ok=True)
    target_labels_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(json_path):
        print(f" Annotation file not found: {json_path}")
        return

    annotation_file = open(json_path, 'r')
    coco_data = json.load(annotation_file)
    annotation_file.close()

    # Build a flat lookup from image ID to its full metadata record
    images_metadata = {}
    for image_data in coco_data['images']:
        images_metadata[image_data['id']] = image_data

    # Group all annotations by their parent image ID for efficient per-image access
    annotations_map = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        annotations_map.setdefault(image_id, []).append(annotation)

    # Collect all image files present on disk for this split
    available_image_files = []
    for filename in os.listdir(images_source_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            available_image_files.append(filename)

    # Build a reverse lookup from filename to image ID to cross-reference disk files with JSON metadata
    filename_to_id_map = {}
    for image_data in coco_data['images']:
        filename_to_id_map[image_data['file_name']] = image_data['id']

    # Filter only images that are both in the folder and in the JSON metadata
    valid_image_files = []
    for filename in available_image_files:
        if filename in filename_to_id_map:
            valid_image_files.append(filename)

    if len(valid_image_files) > max_samples:
        selected_images = random.sample(valid_image_files, max_samples)
    else:
        selected_images = valid_image_files

    processed_count = 0

    for filename in selected_images:
        image_id = filename_to_id_map[filename]
        image_info = images_metadata[image_id]
        image_width = image_info['width']
        image_height = image_info['height']

        yolo_lines = []
        for annotation in annotations_map.get(image_id, []):
            x_min, y_min, box_w, box_h = annotation['bbox']

            # Normalize and clamp coordinates
            x_center = max(0.0, min(1.0, (x_min + (box_w / 2)) / image_width))
            y_center = max(0.0, min(1.0, (y_min + (box_h / 2)) / image_height))
            norm_w = max(0.0, min(1.0, box_w / image_width))
            norm_h = max(0.0, min(1.0, box_h / image_height))

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if yolo_lines:
            # Apply 'rpc_' prefix to both image and label filenames for safety
            target_image_name = f"rpc_{filename}"
            target_label_name = f"rpc_{Path(filename).stem}.txt"

            label_output_file = open(target_labels_path / target_label_name, 'w')
            label_output_file.write("\n".join(yolo_lines))
            label_output_file.close()

            shutil.copy2(os.path.join(images_source_dir, filename), target_images_path / target_image_name)
            processed_count += 1

    print(f"[{split_name.upper()}] Injected {processed_count} RPC images with prefix protection.")


if __name__ == "__main__":
    unified_dir = os.path.join(config.VISION_DATA_DIR, "mvtec_dataset")
    rpc_dir = os.path.join(config.VISION_DATA_DIR, "rpc_raw")

    print(" Injecting RPC TRAIN (Source: test2019_sample) ---")
    convert_and_sample_rpc_with_prefix(
        json_path=os.path.join(rpc_dir, "instances_test2019.json"),
        images_source_dir=os.path.join(rpc_dir, "test2019_sample"),
        target_base_dir=unified_dir,
        split_name="train",
        max_samples=8500
    )

    print("\n Injecting RPC VAL (Source: val2019)")
    convert_and_sample_rpc_with_prefix(
        json_path=os.path.join(rpc_dir, "instances_val2019.json"),
        images_source_dir=os.path.join(rpc_dir, "val2019"),
        target_base_dir=unified_dir,
        split_name="val",
        max_samples=1500
    )

    print("\nHybrid dataset finalized and protected.")
