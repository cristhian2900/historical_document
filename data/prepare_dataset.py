import os
import json
import shutil
from pathlib import Path
import logging
from PIL import Image

class DatasetPreparer:
    def __init__(self, dataset_paths, output_path, image_size=(256, 256)):
        self.dataset_paths = dataset_paths
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            filename=self.output_path / 'dataset_preparation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting dataset preparation")
        print("Starting dataset preparation...")

    def prepare_datasets(self):
        for dataset_path in self.dataset_paths:
            self.logger.info(f"Processing dataset at {dataset_path}")
            print(f"Processing dataset at {dataset_path}...")
            self.process_dataset(Path(dataset_path))

    def process_dataset(self, dataset_path):
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.error(f"Images or labels directory does not exist in {dataset_path}")
            print(f"Error: Images or labels directory does not exist in {dataset_path}")
            return
        
        for image_file in images_dir.glob('*'):
            if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                self.copy_and_resize_image(image_file, labels_dir)

    def copy_and_resize_image(self, image_file, labels_dir):
        image_name = image_file.stem
        label_file = labels_dir / f"{image_name}.txt"
        
        if label_file.exists():
            try:
                with Image.open(image_file) as img:
                    img = img.resize(self.image_size, Image.LANCZOS)
                    img.save(self.output_path / image_file.name)
                shutil.copy2(label_file, self.output_path / label_file.name)
                self.logger.info(f"Copied and resized {image_file.name} and {label_file.name}")
                print(f"Copied and resized {image_file.name} and {label_file.name}")
            except Exception as e:
                self.logger.error(f"Error processing {image_file.name}: {str(e)}")
                print(f"Error processing {image_file.name}: {str(e)}")
        else:
            self.logger.warning(f"No label found for {image_file.name}")
            print(f"No label found for {image_file.name}")

if __name__ == "__main__":
    dataset_paths = [
        "/UE/Master Thesis/dataset_ready/IAM_dataset",
        "/UE/Master Thesis/dataset_ready/icdar_dataset",
        "/UE/Master Thesis/dataset_ready/bentham_dataset",
        "/UE/Master Thesis/dataset_ready/Zenodo"
    ]
    output_path = "/UE/Master Thesis/dataset_ready/prepared_dataset"
    
    preparer = DatasetPreparer(dataset_paths, output_path)
    preparer.prepare_datasets()
    print("Dataset preparation completed.")
