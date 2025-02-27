import os
import json
import pandas as pd
from pathlib import Path
import logging
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET

class ICDARProcessor:
    def __init__(self, raw_dataset_path, normalized_dataset_path):
        self.raw_dataset_path = Path(raw_dataset_path)
        self.normalized_dataset_path = Path(normalized_dataset_path)
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        self.label_extensions = ['.txt', '.json', '.csv', '.xml']
        self.dataset_info = {
            'images': [],
            'labels': {},
            'unmatched_images': []
        }
        
        self.log_dir = self.raw_dataset_path / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "dataset_analysis.log"
        
        self.normalized_dataset_path.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            filename=str(self.log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting ICDAR dataset processing")

    def find_all_files(self):
        self.logger.info("Scanning ICDAR dataset...")
        for root, _, files in os.walk(self.raw_dataset_path):
            for file in files:
                file_path = Path(root) / file
                if any(file_path.name.lower().endswith(ext) for ext in self.image_extensions):
                    self.dataset_info['images'].append(file_path)
                elif any(file_path.name.lower().endswith(ext) for ext in self.label_extensions):
                    self.process_label_file(file_path)
        self.logger.info(f"Found {len(self.dataset_info['images'])} images and {len(self.dataset_info['labels'])} labels")

    def process_label_file(self, file_path):
        if file_path.suffix == '.txt':
            self.process_txt_label(file_path)
        elif file_path.suffix == '.json':
            self.process_json_label(file_path)
        elif file_path.suffix == '.xml':
            self.process_xml_label(file_path)

    def process_txt_label(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            base_name = file_path.stem
            self.dataset_info['labels'][base_name] = content

    def process_json_label(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    self.dataset_info['labels'][key] = str(value)

    def process_xml_label(self, file_path):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for image_elem in root.findall('.//image'):
                image_id = image_elem.get('id', '')
                text_elem = image_elem.find('.//text')
                content = text_elem.text.strip() if text_elem is not None else ''
                if image_id:
                    self.dataset_info['labels'][image_id] = content
        except Exception as e:
            self.logger.error(f"Error reading XML file {file_path}: {str(e)}")

    def match_images_labels(self):
        matched_pairs = []
        for image_path in tqdm(self.dataset_info['images'], desc="Matching"):
            image_id = image_path.stem
            possible_ids = [image_id, image_id.split('.')[0], image_id.lower(), image_id.upper()]
            matched = False
            for possible_id in possible_ids:
                if possible_id in self.dataset_info['labels']:
                    matched_pairs.append({
                        'image': str(image_path),
                        'label_content': self.dataset_info['labels'][possible_id]
                    })
                    matched = True
                    break
            if not matched:
                self.dataset_info['unmatched_images'].append(str(image_path))
        return matched_pairs

    def save_normalized_dataset(self, matched_pairs):
        images_dir = self.normalized_dataset_path / 'images'
        labels_dir = self.normalized_dataset_path / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, pair in enumerate(tqdm(matched_pairs, desc="Saving")):
            image_source = Path(pair['image'])
            image_dest = images_dir / image_source.name
            label_dest = labels_dir / f"{image_source.stem}.txt"
            shutil.copy2(image_source, image_dest)
            with open(label_dest, 'w', encoding='utf-8') as f:
                f.write(pair['label_content'])

        with open(self.normalized_dataset_path / 'matched_pairs.json', 'w') as f:
            json.dump(matched_pairs, f, indent=4)
        
        with open(self.normalized_dataset_path / 'unmatched_images.json', 'w') as f:
            json.dump(self.dataset_info['unmatched_images'], f, indent=4)

    def process(self):
        self.find_all_files()
        matched_pairs = self.match_images_labels()
        self.save_normalized_dataset(matched_pairs)
        
        print(f"\nProcessing complete:")
        print(f"Total images found: {len(self.dataset_info['images'])}")
        print(f"Successfully matched: {len(matched_pairs)}")
        print(f"Unmatched images: {len(self.dataset_info['unmatched_images'])}")
        print(f"Results saved to: {self.normalized_dataset_path}")
        print(f"Log file location: {self.log_file}")

if __name__ == "__main__":
    raw_dataset_path = "/UE/Master Thesis/raw_dataset/zenodo dataset ICDAR 2019/READ-ICDAR2019-cBAD-dataset"
    normalized_dataset_path = "/UE/Master Thesis/dataset_ready/Icdar"
    
    processor = ICDARProcessor(raw_dataset_path, normalized_dataset_path)
    processor.process()