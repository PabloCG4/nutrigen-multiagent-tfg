import os
import shutil
from pathlib import Path
from src.vision import config

class OFFProcessor:
    """
    Handles the reorganization of Open Food Facts images for Fine-Grained Classification.
    TRANSFORMS: Categorical hierarchy (Folder=Category) -> Product hierarchy (Folder=Barcode).
    """

    def __init__(self, input_path: str, output_path: str):
        """
        Args:
            input_path: Path to the raw downloaded images (categories as folders).
            output_path: Path where the cleaner, barcode-based dataset will be created.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.stats = {
            "processed": 0,
            "duplicates": 0
        }

    def run(self):
        """
        Main execution method.
        Iterates through category folders, extracts barcodes, and restructures the dataset.
        """
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
        files_to_process = []
        
        # rglob allows us to search recursively for all image files in the input directory and its subdirectories.
        for ext in extensions:
            files_to_process.extend(list(self.input_path.rglob(ext)))

        for current_image_path in files_to_process:
            self.process_image(current_image_path)

        self.show_summary()

    def process_image(self, file_path: Path):
        """
        Moves a single image to its barcode-specific folder.
        """
        # Extract Barcode from filename
        barcode = file_path.stem
        extension = file_path.suffix

        # Basic validation: Barcodes must be numeric and have a minimal length
        if not barcode.isdigit() or len(barcode) < 3:
            return

        # Define destination paths
        # Structure: Output / Barcode / Barcode.jpg
        product_folder = self.output_path / barcode
        destination_file = product_folder / f"{barcode}{extension}"

        # Prevent data leakage by handling duplicates
        if destination_file.exists():
            self.stats["duplicates"] += 1
            return

        # Create the specific product folder and copy the file
        product_folder.mkdir(exist_ok=True)
        shutil.copy2(file_path, destination_file)
        
        self.stats["processed"] += 1

    def show_summary(self):
        unique_products = len(list(self.output_path.iterdir()))
        
        print(f"Total Unique Products: {unique_products}")
        print(f"Images Organized: {self.stats['processed']}")
        print(f"Duplicates Skipped: {self.stats['duplicates']}")

if __name__ == "__main__":
    clean_dataset = os.path.join(config.DATA_DIR, "vision", "classification_dataset")

    processor = OFFProcessor(config.IMAGES_DIR, clean_dataset)
    processor.run()