import os
import torch
from torch.utils.data import Dataset, DataLoader
import chromadb
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.vision import config
from src.vision.processing.transforms import SquarePadding


class ProductBaselineDataset(Dataset):
    """
    Dataset class optimized for inference and vector extraction.
    Handles parallel image loading and applies Letterbox transformations.
    Requires a strict folder structure where the directory name is the unique barcode.

    This non-augmentation baseline stores exactly one vector per source image,
    without modifying the underlying pixels (no flips/rotations/brightness changes).
    """

    def __init__(self, dataset_directory):
        self.dataset_path = Path(dataset_directory)
        self.items_to_process = []

        # Standard ImageNet normalization pipeline required by DINOv2 architecture
        # SquarePadding is added here to preserve aspect ratio before resizing.
        self.image_transform = transforms.Compose([
            SquarePadding(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.gather_dataset_items()

    def gather_dataset_items(self):
        """
        Scans the directory structure to build an index of available images.
        Assumes the parent directory name acts as the ground-truth barcode.

        Unique IDs follow the pattern "{barcode}_{image_index}_0".
        The suffix is kept for compatibility with older augmented IDs.
        """
        for folder_path in self.dataset_path.iterdir():
            if folder_path.is_dir():
                barcode = folder_path.name
                image_index = 0

                for file_path in folder_path.iterdir():
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.items_to_process.append({
                            "unique_id": f"{barcode}_{image_index}_0",
                            "barcode": barcode,
                            "image_path": str(file_path),
                        })
                        image_index += 1

    def __len__(self):
        return len(self.items_to_process)

    def __getitem__(self, index):
        """
        Retrieves and transforms a single image on demand.
        Returns a 4-tuple: (tensor, unique_id, barcode, image_path).

        No offline augmentation is applied here; the only pixel-space operation
        is deterministic preprocessing (SquarePadding + normalization) required
        by the DINOv2 model.
        """
        item = self.items_to_process[index]
        image = Image.open(item["image_path"]).convert("RGB")

        tensor_image = self.image_transform(image)
        return tensor_image, item["unique_id"], item["barcode"], item["image_path"]


class BaselineDatabasePopulator:
    """
    Orchestrates the extraction of high-dimensional embeddings using the raw DINOv2
    baseline model, persisting them efficiently in a ChromaDB vector store.
    """

    def __init__(self, batch_size=64, computation_device="cuda"):
        self.batch_size = batch_size
        self.active_device = computation_device
        self.collection_name = "product_embeddings"

        print(" Loading raw DINOv2 model (Baseline mode)...")

        # Initialize the base Vision Transformer model
        self.vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.vision_encoder.to(self.active_device)
        self.vision_encoder.eval()

        # Initialize the Persistent Vector Database Client
        database_path_string = str(Path(config.DATA_DIR) / "chroma_db_store" / "chroma_db_raw_baseline")
        print(f" Connecting to baseline ChromaDB at: {database_path_string}")
        self.vector_database_client = chromadb.PersistentClient(path=database_path_string)

        # Configure the collection to use Cosine Similarity for the HNSW index
        self.product_collection = self.vector_database_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def populate_database(self, dataset_directory):
        """
        Executes the batch processing pipeline to extract features and populate the vector store.
        """
        dataset = ProductBaselineDataset(dataset_directory)

        # Utilize hardware-accelerated dataloading to maximize throughput
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        total_items = len(dataset)
        processed_count = 0

        # Buffer configuration to optimize disk write operations
        accumulation_limit = 1000
        pending_embeddings = []
        pending_ids = []
        pending_metadata = []

        print(f" Starting population of {total_items} images...")

        # Disable gradient computation to free up VRAM and accelerate inference
        with torch.no_grad():
            for tensor_batch, unique_id_batch, barcode_batch, path_batch in data_loader:
                # Transfer data to the target compute device asynchronously
                tensor_batch = tensor_batch.to(self.active_device, non_blocking=True)

                # Feature Extraction Pipeline
                raw_feature_vectors = self.vision_encoder(tensor_batch)

                # Normalize feature vectors to length 1.0 to ensure correct Cosine Similarity calculation
                l2_normalized_vectors = torch.nn.functional.normalize(raw_feature_vectors, p=2, dim=1)

                # Detach from VRAM and convert to native Python lists for the database API
                embeddings_list = l2_normalized_vectors.cpu().numpy().tolist()

                # Accumulate embeddings and metadata in memory until the threshold is reached
                pending_embeddings.extend(embeddings_list)

                # unique_id_batch (e.g. "8410188011015_0") is used as the ChromaDB document ID
                # so multiple images of the same product coexist without ID collision.
                pending_ids.extend(unique_id_batch)

                # Construct metadata entries for each product associating barcode and image path
                for path, barcode in zip(path_batch, barcode_batch):
                    pending_metadata.append({"barcode": barcode, "path": path})

                processed_count += len(barcode_batch)

                # Execute batch insert to the database once the buffer is full
                if len(pending_ids) >= accumulation_limit:
                    self.product_collection.add(
                        ids=pending_ids,
                        embeddings=pending_embeddings,
                        metadatas=pending_metadata
                    )

                    # Reset buffers
                    pending_embeddings.clear()
                    pending_ids.clear()
                    pending_metadata.clear()
                    print(f" Inserted into database. Progress: {processed_count}/{total_items}")

        # Flush any remaining items in the buffer
        if pending_ids:
            self.product_collection.add(
                ids=pending_ids,
                embeddings=pending_embeddings,
                metadatas=pending_metadata
            )
            print(f" Final insertion complete. Total populated: {processed_count}/{total_items}")


if __name__ == "__main__":
    populator = BaselineDatabasePopulator(batch_size=64, computation_device="cuda")
    populator.populate_database(config.CLASSIFICATION_DATASET_DIR_ORIGINAL)

