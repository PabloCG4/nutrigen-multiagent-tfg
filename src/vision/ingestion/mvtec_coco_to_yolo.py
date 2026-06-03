import json
import os
from pathlib import Path
from src.vision import config


def convert_coco_to_yolo_single_class(coco_json_path, output_directory):
    """
    Parses a COCO format JSON annotation file and converts the bounding boxes
    to the YOLO format. It intentionally overrides all original class IDs 
    and forces every object to be class 0 ('product'), facilitating an 
    agnostic objectness detection strategy.
    """
    print("Loading COCO JSON annotations...")

    json_file = open(coco_json_path, 'r', encoding='utf-8')
    coco_data = json.load(json_file)
    json_file.close()

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Establish a mapping to quickly retrieve image dimensions using their ID
    image_metadata_mapping = {}
    for image_info in coco_data['images']:
        image_metadata_mapping[image_info['id']] = {
            'file_name': image_info['file_name'],
            'width': float(image_info['width']),
            'height': float(image_info['height'])
        }

    yolo_annotations_dictionary = {}
    processed_boxes_count = 0

    print("Converting bounding box coordinates...")

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']

        if image_id not in image_metadata_mapping:
            continue

        image_data = image_metadata_mapping[image_id]
        image_width = image_data['width']
        image_height = image_data['height']

        # COCO bounding box format is [top_left_x, top_left_y, width, height] in absolute pixels
        absolute_x_minimum, absolute_y_minimum, box_width, box_height = annotation['bbox']

        # YOLO expects normalized coordinates (0.0 to 1.0) for the center of the box, plus normalized width and height
        center_x = (absolute_x_minimum + (box_width / 2.0)) / image_width
        center_y = (absolute_y_minimum + (box_height / 2.0)) / image_height
        normalized_width = box_width / image_width
        normalized_height = box_height / image_height

        # Mathematical safeguard to clamp values and prevent YOLO out-of-bounds dataset errors
        # which can crash the training process during data loading
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        normalized_width = max(0.0, min(1.0, normalized_width))
        normalized_height = max(0.0, min(1.0, normalized_height))

        # Force class 0 for all instances
        forced_class_id = 0
        yolo_formatted_string = f"{forced_class_id} {center_x:.6f} {center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}"

        if image_id not in yolo_annotations_dictionary:
            yolo_annotations_dictionary[image_id] = []

        yolo_annotations_dictionary[image_id].append(yolo_formatted_string)
        processed_boxes_count += 1

    print(f"Writing YOLO annotation files to {output_directory}...")

    for image_id, bounding_boxes_list in yolo_annotations_dictionary.items():
        original_file_name = image_metadata_mapping[image_id]['file_name']
        base_name_without_extension = os.path.splitext(original_file_name)[0]
        yolo_text_filename = f"{base_name_without_extension}.txt"
        yolo_text_filepath = os.path.join(output_directory, yolo_text_filename)

        text_file = open(yolo_text_filepath, 'w', encoding='utf-8')
        text_file.write("\n".join(bounding_boxes_list))
        text_file.close()

    total_images_processed = len(yolo_annotations_dictionary)
    print(f"Operation completed. Successfully exported {processed_boxes_count} objects across {total_images_processed} images.")


if __name__ == "__main__":
    INPUT_MVTEC_JSON_FILE = r"C:\Users\Usuario\OneDrive\Escritorio\TFG\data\vision\mvtec_raw\annotations\D2S_training.json"
    convert_coco_to_yolo_single_class(INPUT_MVTEC_JSON_FILE, config.OUTPUT_MVTEC_TXT_DIRECTORY)
