import os
from pathlib import Path
import logging
import pandas as pd
from PIL import Image
import json
import shutil
from tqdm import tqdm
from datetime import datetime

class DatasetValidator:
    def __init__(self, dataset_path: Path, output_path: Path, mapping_file: Path):
        self.dataset_path = Path(dataset_path)
        self.output_path = output_path
        self.mapping_file = mapping_file
        self.logger = self._setup_logging()
        
        # Statistics
        self.valid_pairs = 0
        self.invalid_images = 0
        self.invalid_labels = 0
        self.skipped_pairs = 0
        
        # Debug information
        self.invalid_image_files = []
        self.invalid_label_content = []
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.output_path / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"dataset_validation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _is_valid_image(self, image_path: Path) -> bool:
        """Check if an image file is valid"""
        try:
            with Image.open(image_path) as img:
                img.verify()
                # Additional checks if needed
                if img.size[0] < 10 or img.size[1] < 10:  # Minimum size check
                    self.invalid_image_files.append(f"{image_path} (Invalid size: {img.size})")
                    return False
                return True
        except Exception as e:
            self.invalid_image_files.append(f"{image_path} (Error: {str(e)})")
            return False

    def _is_valid_label(self, label_path: Path) -> bool:
        """Check if a label file is valid"""
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if the content is not empty
                if not content:
                    self.invalid_label_content.append(f"{label_path} (Empty file)")
                    return False
                return True
        except Exception as e:
            self.invalid_label_content.append(f"{label_path} (Error: {str(e)})")
            return False

    def validate_and_clean(self):
        """Validate the dataset using the mapping file"""
        self.logger.info(f"Starting dataset validation from: {self.dataset_path}")
        self.logger.info(f"Output directory: {self.output_path}")

        # Create output directories
        images_out = self.output_path / 'images'
        labels_out = self.output_path / 'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Load the mapping file
        try:
            mapping_df = pd.read_csv(self.mapping_file)
            self.logger.info(f"Loaded mapping file with {len(mapping_df)} entries")
        except Exception as e:
            self.logger.error(f"Error loading mapping file: {str(e)}")
            return

        # Process each image-label pair
        for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Validating pairs"):
            try:
                image_path = self.dataset_path / row['image']
                label_path = self.dataset_path / row['label']

                # Skip if either file doesn't exist
                if not image_path.exists() or not label_path.exists():
                    self.skipped_pairs += 1
                    continue

                # Validate image
                if not self._is_valid_image(image_path):
                    self.invalid_images += 1
                    continue

                # Validate label
                if not self._is_valid_label(label_path):
                    self.invalid_labels += 1
                    continue

                # Both are valid, copy to output directory
                try:
                    # Create necessary subdirectories in output
                    new_image_path = images_out / image_path.name
                    new_label_path = labels_out / label_path.name
                    
                    # Copy files
                    shutil.copy2(image_path, new_image_path)
                    shutil.copy2(label_path, new_label_path)
                    
                    self.valid_pairs += 1
                    
                except Exception as e:
                    self.logger.error(f"Error copying files: {str(e)}")
                    continue

            except Exception as e:
                self.logger.error(f"Error processing pair: {str(e)}")
                continue

        # Save validation results and log statistics
        self._save_validation_results()
        self._log_statistics()

    def _save_validation_results(self):
        """Save validation results to a JSON file"""
        results = {
            'valid_pairs': self.valid_pairs,
            'invalid_images': self.invalid_images,
            'invalid_labels': self.invalid_labels,
            'skipped_pairs': self.skipped_pairs,
            'invalid_image_files': self.invalid_image_files[:100],
            'invalid_label_content': self.invalid_label_content[:100]
        }
        
        results_file = self.output_path / 'validation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    def _log_statistics(self):
        """Log the validation statistics"""
        self.logger.info("\n=== Validation Statistics ===")
        self.logger.info(f"Valid pairs processed: {self.valid_pairs}")
        self.logger.info(f"Invalid images: {self.invalid_images}")
        self.logger.info(f"Invalid labels: {self.invalid_labels}")
        self.logger.info(f"Skipped pairs: {self.skipped_pairs}")
        self.logger.info("==========================")

def main():
    # Define paths
    dataset_path = Path("/UE/Master Thesis/dataset_ready")
    output_path = Path("/UE/Master Thesis/dataset_ready/prepared_dataset")
    mapping_file = Path("/UE/Master Thesis/mapping.csv")
    
    try:
        validator = DatasetValidator(dataset_path, output_path, mapping_file)
        validator.validate_and_clean()
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 