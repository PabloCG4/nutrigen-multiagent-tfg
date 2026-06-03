import os
import shutil
import random
from typing import List, Dict

from src.vision import config


def create_directory_structure(base_output_path: str) -> None:
    """Creates the standard YOLO directory structure for training, validation, and testing."""
    dataset_splits = ['train', 'valid', 'test']
    for current_split in dataset_splits:
        os.makedirs(os.path.join(base_output_path, current_split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_output_path, current_split, 'labels'), exist_ok=True)


def process_standard_datasets(raw_base_path: str, output_base_path: str, dataset_names: List[str]) -> None:
    """Processes standard YOLO formatted datasets, forcing the class ID to 0."""
    dataset_splits = ['train', 'valid', 'test']

    # Iterate through the list of dataset names
    for dataset_name in dataset_names:
        # Iterate through the list of splits
        for current_split in dataset_splits:
            # Get the input images and labels directories
            input_images_directory = os.path.join(raw_base_path, dataset_name, current_split, 'images')
            input_labels_directory = os.path.join(raw_base_path, dataset_name, current_split, 'labels')

            # Get the output images and labels directories
            output_images_directory = os.path.join(output_base_path, current_split, 'images')
            output_labels_directory = os.path.join(output_base_path, current_split, 'labels')

            # Check if the input images and labels directories exist
            if not os.path.exists(input_images_directory) or not os.path.exists(input_labels_directory):
                continue

            # Iterate through the list of image files in the input images directory
            for image_file_name in os.listdir(input_images_directory):
                # Get the base filename and label file name
                base_filename = os.path.splitext(image_file_name)[0]
                label_file_name = base_filename + '.txt'

                # Get the input image and label file paths
                input_image_path = os.path.join(input_images_directory, image_file_name)
                input_label_path = os.path.join(input_labels_directory, label_file_name)

                if os.path.exists(input_label_path):
                    # Generate the new image and label file names
                    new_image_filename = f"{dataset_name}_{image_file_name}"
                    new_label_filename = f"{dataset_name}_{label_file_name}"

                    # Copy the input image to the output images directory
                    shutil.copy2(input_image_path, os.path.join(output_images_directory, new_image_filename))

                    # Read the input label file. We need to insert a 0 in the first position of the line to force the class ID to 0.
                    read_file_handler = open(input_label_path, 'r')
                    try:
                        original_lines = read_file_handler.readlines()
                    finally:
                        read_file_handler.close()

                    # Process the original lines to force the class ID to 0
                    processed_lines = []
                    for line in original_lines:
                        line_components = line.strip().split()
                        if len(line_components) >= 5:
                            line_components[0] = '0'
                            processed_lines.append(' '.join(line_components) + '\n')

                    # Write the processed lines to the output label file
                    output_label_path = os.path.join(output_labels_directory, new_label_filename)
                    write_file_handler = open(output_label_path, 'w')
                    try:
                        write_file_handler.writelines(processed_lines)
                    finally:
                        write_file_handler.close()


def convert_sku_annotations_to_yolo(annotations_file_path: str) -> Dict[str, List[str]]:
    """Parses SKU CSV annotations and converts them to YOLO format grouped by image filename.
    We need to insert a 0 in the first position of the line to force the class ID to 0.
    SKU format is: 
    image_name, xmin_string, ymin_string, xmax_string, ymax_string, class_name, width_string, height_string
    We need to convert this to YOLO format:
    0 x_center_normalized y_center_normalized width_normalized height_normalized
    """
    image_annotations_map = {}

    read_file_handler = open(annotations_file_path, 'r')
    try:
        for line in read_file_handler:
            line_components = line.strip().split(',')
            if len(line_components) != 8:
                continue

            image_name, xmin_string, ymin_string, xmax_string, ymax_string, class_name, width_string, height_string = line_components

            try:
                xmin_coordinate = float(xmin_string)
                ymin_coordinate = float(ymin_string)
                xmax_coordinate = float(xmax_string)
                ymax_coordinate = float(ymax_string)
                image_width = float(width_string)
                image_height = float(height_string)
            except ValueError:
                continue

            bounding_box_width = xmax_coordinate - xmin_coordinate
            bounding_box_height = ymax_coordinate - ymin_coordinate
            x_center_coordinate = xmin_coordinate + (bounding_box_width / 2.0)
            y_center_coordinate = ymin_coordinate + (bounding_box_height / 2.0)

            x_center_normalized = x_center_coordinate / image_width
            y_center_normalized = y_center_coordinate / image_height
            width_normalized = bounding_box_width / image_width
            height_normalized = bounding_box_height / image_height

            yolo_formatted_line = f"0 {x_center_normalized:.6f} {y_center_normalized:.6f} {width_normalized:.6f} {height_normalized:.6f}\n"

            if image_name not in image_annotations_map:
                image_annotations_map[image_name] = []
            image_annotations_map[image_name].append(yolo_formatted_line)
    finally:
        read_file_handler.close()

    return image_annotations_map


def process_sku_dataset(raw_base_path: str, output_base_path: str, target_samples_limit: int = 1500) -> None:
    """Processes a specific subset of the SKU dataset and converts annotations to YOLO format."""
    sku_base_directory = os.path.join(raw_base_path, 'sku_dataset')
    annotations_directory = os.path.join(sku_base_directory, 'anotations')
    images_directory = os.path.join(sku_base_directory, 'images')

    # We are not using the entire SKU dataset, we are only using a subset of the annotations.
    splits_distribution_map = {
        'train': {'file_name': 'anotations_train', 'ratio_proportion': 0.8},
        'valid': {'file_name': 'anotations_val', 'ratio_proportion': 0.1},
        'test': {'file_name': 'anotations_test', 'ratio_proportion': 0.1}
    }

    for current_split, split_configuration in splits_distribution_map.items():
        annotation_file_path = os.path.join(annotations_directory, split_configuration['file_name'])
        if not os.path.exists(annotation_file_path):
            continue

        image_annotations_map = convert_sku_annotations_to_yolo(annotation_file_path)
        available_images_list = list(image_annotations_map.keys())

        split_target_samples = int(target_samples_limit * split_configuration['ratio_proportion'])

        # If the number of available images is greater than the target number of samples, we select a random sample of the available images.
        if len(available_images_list) > split_target_samples:
            selected_images_list = random.sample(available_images_list, split_target_samples)
        else:
            selected_images_list = available_images_list

        output_images_directory = os.path.join(output_base_path, current_split, 'images')
        output_labels_directory = os.path.join(output_base_path, current_split, 'labels')

        for image_name in selected_images_list:
            source_image_path = os.path.join(images_directory, image_name)

            if os.path.exists(source_image_path):
                new_image_filename = f"sku_{image_name}"
                shutil.copy2(source_image_path, os.path.join(output_images_directory, new_image_filename))
                
                # The base filename is the filename without the extension. splitext returns a tuple with the filename and the extension.
                base_filename = os.path.splitext(new_image_filename)[0]
                label_output_path = os.path.join(output_labels_directory, base_filename + '.txt')

                write_file_handler = open(label_output_path, 'w')
                try:
                    write_file_handler.writelines(image_annotations_map[image_name])
                finally:
                    write_file_handler.close()


def process_negative_dataset(raw_base_path: str, output_base_path: str) -> None:
    """Processes negative images by allocating them to splits and generating empty label files."""
    negatives_directory = os.path.join(raw_base_path, 'negatives')
    if not os.path.exists(negatives_directory):
        return

    # Collect all valid image files from the negatives directory
    all_negative_images = []
    for file_name in os.listdir(negatives_directory):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_negative_images.append(file_name)

    random.shuffle(all_negative_images)

    total_images_count = len(all_negative_images)
    train_split_end_index = int(total_images_count * 0.8)
    valid_split_end_index = train_split_end_index + int(total_images_count * 0.1)

    splits_allocation_map = {
        'train': all_negative_images[:train_split_end_index],
        'valid': all_negative_images[train_split_end_index:valid_split_end_index],
        'test': all_negative_images[valid_split_end_index:]
    }

    for current_split, allocated_images in splits_allocation_map.items():
        output_images_directory = os.path.join(output_base_path, current_split, 'images')
        output_labels_directory = os.path.join(output_base_path, current_split, 'labels')

        for image_name in allocated_images:
            source_image_path = os.path.join(negatives_directory, image_name)
            new_image_filename = f"negative_{image_name}"

            shutil.copy2(source_image_path, os.path.join(output_images_directory, new_image_filename))

            base_filename = os.path.splitext(new_image_filename)[0]
            label_output_path = os.path.join(output_labels_directory, base_filename + '.txt')

            write_file_handler = open(label_output_path, 'w')
            try:
                write_file_handler.write("")
            finally:
                write_file_handler.close()


def assemble_master_dataset() -> None:
    """Main execution flow to assemble the final unified dataset."""
    random.seed(42)

    raw_dataset_directory = config.DETECTION_DATASET_RAW_DIR
    final_dataset_directory = os.path.join(config.DATA_DIR, "vision", "dataset_frontal_unificado")

    create_directory_structure(final_dataset_directory)

    standard_datasets_list = ['fridge', 'shelf', 'shelf2']
    process_standard_datasets(raw_dataset_directory, final_dataset_directory, standard_datasets_list)

    process_sku_dataset(raw_dataset_directory, final_dataset_directory, target_samples_limit=1500)

    process_negative_dataset(raw_dataset_directory, final_dataset_directory)


if __name__ == "__main__":
    assemble_master_dataset()
