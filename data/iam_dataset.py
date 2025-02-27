import os
import json
import csv
import pandas as pd
from pathlib import Path
import logging
import shutil
from tqdm import tqdm

class IAMDatasetProcessor:
    def __init__(self, raw_dataset_path, normalized_dataset_path):
        self.raw_dataset_path = Path(raw_dataset_path)
        self.normalized_dataset_path = Path(normalized_dataset_path)
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        self.label_extensions = ['.txt', '.json', '.csv', '.xlsx']
        self.dataset_info = {
            'images': [],
            'labels': {},
            'unmatched_images': []
        }
    
        self.log_dir = self.raw_dataset_path / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "dataset_analysis.log"
        logging.basicConfig(
            filename=str(self.log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def find_all_label_files(self):
        print("Scanning for label files...")
        label_files = []
        
        for ext in self.label_extensions:
            found_files = list(self.raw_dataset_path.rglob(f'*{ext}'))
            label_files.extend(found_files)
            print(f"Found {len(found_files)} {ext} files")
        
        return label_files

    def process_txt_label(self, file_path):
        labels = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        image_id, text = parts
                        labels[image_id] = text
        except Exception as e:
            self.logger.error(f"Error processing TXT file {file_path}: {str(e)}")
        return labels

    def process_json_label(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                else:
                    self.logger.warning(f"Unexpected JSON structure in {file_path}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error processing JSON file {file_path}: {str(e)}")
            return {}

    def process_csv_excel_label(self, file_path):
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:  
                df = pd.read_excel(file_path)
            
            if 'image_id' in df.columns and 'text' in df.columns:
                return dict(zip(df['image_id'], df['text']))
            else:
                self.logger.warning(f"Required columns not found in {file_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error processing {file_path.suffix} file {file_path}: {str(e)}")
            return {}

    def process_label_files(self):

        label_files = self.find_all_label_files()
        
        for label_file in tqdm(label_files, desc="Processing label files"):
            try:
                if label_file.suffix == '.txt':
                    labels = self.process_txt_label(label_file)
                elif label_file.suffix == '.json':
                    labels = self.process_json_label(label_file)
                elif label_file.suffix in ['.csv', '.xlsx']:
                    labels = self.process_csv_excel_label(label_file)
                
                self.dataset_info['labels'].update(labels)
                
            except Exception as e:
                self.logger.error(f"Error processing {label_file}: {str(e)}")

    def find_all_images(self):

        print("Scanning for images...")
        for ext in self.image_extensions:
            found_images = list(self.raw_dataset_path.rglob(f'*{ext}'))
            self.dataset_info['images'].extend(found_images)
            print(f"Found {len(found_images)} {ext} images")

    def match_and_save(self):

        print("Matching images with labels...")
        
        images_dir = self.normalized_dataset_path / 'images'
        labels_dir = self.normalized_dataset_path / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        matched_pairs = []
        unmatched_images = []
        
        for image_path in tqdm(self.dataset_info['images'], desc="Processing images"):
            image_id = image_path.stem
            
            possible_ids = [
                image_id,
                image_id.split('-')[0],
                image_id.lower(),
                image_id.upper()
            ]
            
            matched = False
            for possible_id in possible_ids:
                if possible_id in self.dataset_info['labels']:
                    matched = True
                    shutil.copy2(image_path, images_dir / image_path.name)
                    
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    with open(label_path, 'w', encoding='utf-8') as f:
                        f.write(self.dataset_info['labels'][possible_id])
                    
                    matched_pairs.append({
                        'image': str(image_path),
                        'label': str(label_path)
                    })
                    break
            
            if not matched:
                unmatched_images.append(str(image_path))
        
        with open(self.normalized_dataset_path / 'matched_pairs.json', 'w') as f:
            json.dump(matched_pairs, f, indent=4)
        
        with open(self.normalized_dataset_path / 'unmatched_images.json', 'w') as f:
            json.dump(unmatched_images, f, indent=4)
        
        return len(matched_pairs), len(unmatched_images)

    def process(self):

        print("Starting IAM dataset processing...")
        
        self.process_label_files()
        print(f"Found {len(self.dataset_info['labels'])} labels")
        
        self.find_all_images()
        print(f"Found {len(self.dataset_info['images'])} images")
        
        matched, unmatched = self.match_and_save()
        print(f"\nProcessing complete:")
        print(f"Matched pairs: {matched}")
        print(f"Unmatched images: {unmatched}")
        print(f"Results saved to: {self.normalized_dataset_path}")

if __name__ == "__main__":
    raw_dataset_path = "/UE/Master Thesis/raw_dataset/IAM dataset"
    normalized_dataset_path = "/UE/Master Thesis/dataset_ready/IAM_dataset"
    
    processor = IAMDatasetProcessor(raw_dataset_path, normalized_dataset_path)
    processor.process()