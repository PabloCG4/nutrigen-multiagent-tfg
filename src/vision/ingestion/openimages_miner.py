import os
import glob
from typing import List

import fiftyone as fo
import fiftyone.zoo as foz

from src.vision import config

class OpenImagesMiner:
    """
    Handles the ingestion of the OpenImages V7 dataset for the Object Detection module.
    
    Architecture:
    1. Downloads specific food/container classes from OpenImages.
    2. Exports them to YOLOv8 format keeping original labels to preserve bounding boxes.
    3. Post-processes the labels in-place to unify all classes into a single 'Pantry_Item' (ID 0).
    """

    def __init__(self, maximum_samples_limit: int = 20000):
        """
        Initializes the miner with download constraints.
        
        Args:
            maximum_samples_limit (int): Maximum number of samples to download.
        """
        self.maximum_samples_limit = maximum_samples_limit

    def get_target_classes(self) -> List[str]:
        """
        Returns the specific list of classes to download.
        These classes have been verified against the OpenImages V7 index.
        """
        # 1. Raw Ingredients
        produce_ingredients = [
            "Apple", "Artichoke", "Banana", "Bell pepper", "Broccoli", 
            "Cabbage", "Cantaloupe", "Carrot", "Coconut", "Cucumber", 
            "Garden Asparagus", "Grape", "Grapefruit", "Lemon", "Mango", 
            "Mushroom", "Orange", "Peach", "Pear", "Pineapple", "Pomegranate", 
            "Potato", "Pumpkin", "Radish", "Strawberry", "Tomato", 
            "Watermelon", "Zucchini", "Squash (Plant)"
        ]
        
        # 2. Proteins & Dairy
        protein_and_dairy_sources = [
            "Cheese", "Chicken", "Crab", "Egg (Food)", "Fish", 
            "Hamburger", "Hot dog", "Lobster", "Milk", "Oyster", 
            "Shellfish", "Shrimp", "Sushi"
        ]
        
        # 3. Bakery, Carbs & Prepared Food
        bakery_and_prepared_foods = [
            "Bagel", "Bread", "Cake", "Cookie", "Croissant", 
            "Doughnut", "French fries", "Guacamole", "Ice cream", "Muffin", 
            "Pancake", "Pasta", "Pizza", "Popcorn", "Pretzel", 
            "Salad", "Sandwich", "Taco", "Tart", "Waffle"
        ]
        
        # Removed drinks and bottles

        combined_target_classes = produce_ingredients + protein_and_dairy_sources + bakery_and_prepared_foods 
        combined_target_classes.sort()
        return combined_target_classes

    def flatten_labels_to_single_class(self, dataset_directory_path: str):
        """
        Iterates through all exported YOLO .txt label files and forces the class ID to '0'.
        """
        
        # Pattern to find all .txt files in subdirectories (train/val), ** allows recursive search and *.txt filters for label files
        search_pattern_expression = os.path.join(dataset_directory_path, "**", "*.txt")
        annotation_file_paths = glob.glob(search_pattern_expression, recursive=True)
        
        modified_files_count = 0
        for current_file_path in annotation_file_paths:
            # Skip valid classes file or dataset configuration if accidentally caught
            if current_file_path.endswith("classes.txt"):
                continue

            # Read phase with explicit file handling
            read_file_handler = open(current_file_path, 'r')
            try:
                original_lines = read_file_handler.readlines()
            finally:
                read_file_handler.close()
            
            updated_lines_buffer = []
            for current_line in original_lines:
                line_components = current_line.strip().split()
                # Ensure line has at least class_id + 4 coordinates (x_center, y_center, width, height)
                if len(line_components) >= 5:
                    # Force class ID (index 0) to '0'
                    line_components[0] = "0" 
                    updated_lines_buffer.append(" ".join(line_components) + "\n")
            
            # Write phase, starts with empty file to ensure clean overwrite, then writes all updated lines at once
            if updated_lines_buffer:
                write_file_handler = open(current_file_path, 'w')
                try:
                    write_file_handler.writelines(updated_lines_buffer)
                finally:
                    write_file_handler.close()
                modified_files_count += 1
                
        print(f"Successfully flattened labels in {modified_files_count} files.")

    def run(self):
        """Executes the download, export, and unification process."""
        target_object_classes = self.get_target_classes()
        
        print("Starting FiftyOne ingestion process.")

        # Load the dataset subset
        downloaded_dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=target_object_classes,
            max_samples=self.maximum_samples_limit,
            shuffle=True,
            seed=42
        )

        # Determine correct field name (Robustness check)
        ground_truth_label_field = "ground_truth"
        
        print(f"Exporting raw data from field: '{ground_truth_label_field}'...")

        # Export with ORIGINAL classes first.
        # This prevents FiftyOne from generating empty files due to mapping errors.
        downloaded_dataset.export(
            export_dir=config.YOLO_DATASET_DIR,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=ground_truth_label_field,
            classes=target_object_classes, 
            overwrite=True
        )

        # Flatten labels to ID 0
        self.flatten_labels_to_single_class(config.YOLO_DATASET_DIR)
        
        print("Ingestion, export, and class unification finished successfully.")

if __name__ == "__main__":
    dataset_miner_instance = OpenImagesMiner(maximum_samples_limit=20000)
    dataset_miner_instance.run()