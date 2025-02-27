import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import shutil
from tqdm import tqdm

class IAMHistoricalProcessor:
    def __init__(self, raw_dataset_path, normalized_dataset_path):
        self.raw_dataset_path = Path(raw_dataset_path)
        self.normalized_dataset_path = Path(normalized_dataset_path)
        self.image_extensions = ['.png', '.jpg', '.jpeg']
        self.label_extensions = ['.xml', '.txt']
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
        self.logger.info("Logging initialized successfully")

    def parse_xml_file(self, xml_file):
        labels = {}
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for line in root.findall(".//line"):
                image_id = line.get('id')
                label_text = line.get('text')
                if image_id and label_text:
                    labels[image_id] = label_text.strip()
        except Exception as e:
            self.logger.error(f"Error parsing XML {xml_file}: {str(e)}")
        return labels

    def parse_txt_file(self, txt_file):
        labels = {}
        try:
            with open(txt_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) > 8:
                        line_id = parts[0]
                        label_text = " ".join(parts[8:]).replace("|", " ")
                        labels[line_id] = label_text.strip()
        except Exception as e:
            self.logger.error(f"Error parsing txt {txt_file}: {str(e)}")
        return labels

    def scan_dataset(self):
        self.logger.info("Scanning dataset...")
        
        for root, _, files in os.walk(self.raw_dataset_path):
            self.logger.info(f"Scanning directory: {root}")
            for file in tqdm(files, desc="Scanning files"):
                file_path = Path(root) / file
                if file_path.suffix in self.label_extensions:
                    self.logger.info(f"Found label file: {file_path}")
                    if file_path.suffix == '.xml':
                        labels = self.parse_xml_file(file_path)
                    elif file_path.suffix == '.txt':
                        labels = self.parse_txt_file(file_path)
                    self.logger.info(f"Extracted {len(labels)} labels from {file_path}")
                    self.dataset_info['labels'].update(labels)

        sample_labels = list(self.dataset_info['labels'].items())[:5]
        self.logger.info("Sample labels:")
        for id_, text in sample_labels:
            self.logger.info(f"ID: {id_}, Text: {text}")

        for root, _, files in os.walk(self.raw_dataset_path):
            self.logger.info(f"Scanning directory: {root}")
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    image_path = Path(root) / file
                    self.dataset_info['images'].append(image_path)
                    if len(self.dataset_info['images']) <= 5:
                        self.logger.info(f"Sample image: {image_path}")

        self.logger.info(f"Found {len(self.dataset_info['images'])} images")
        self.logger.info(f"Found {len(self.dataset_info['labels'])} labels")

    def extract_image_id(self, image_path):
        return image_path.stem

    def match_images_labels(self):
        matched_data = []
        unmatched_images = set(self.dataset_info['images'])

        for image_path in tqdm(self.dataset_info['images'], desc="Matching"):
            image_id = self.extract_image_id(image_path)
            if image_id in self.dataset_info['labels']:
                matched_data.append({
                    'image_path': str(image_path),
                    'label': self.dataset_info['labels'][image_id]
                })
                unmatched_images.discard(image_path)
            else:
                for label_id in self.dataset_info['labels']:
                    if image_id in label_id or label_id in image_id:
                        matched_data.append({
                            'image_path': str(image_path),
                            'label': self.dataset_info['labels'][label_id]
                        })
                        unmatched_images.discard(image_path)
                        self.logger.info(f"Partial match found: {image_path.name} with label {label_id}")
                        break

        self.dataset_info['unmatched_images'] = list(unmatched_images)
        return matched_data

    def save_normalized_dataset(self, matched_data):
        images_dir = self.normalized_dataset_path / 'images'
        labels_dir = self.normalized_dataset_path / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for pair in tqdm(matched_data, desc="Saving matched pairs"):
            try:
                image_source = Path(pair['image_path'])
                label_content = pair['label']
                image_dest = images_dir / image_source.name
                shutil.copy2(image_source, image_dest)

                label_dest = labels_dir / f"{image_source.stem}.txt"
                with open(label_dest, 'w', encoding='utf-8') as f:
                    f.write(label_content)
            except Exception as e:
                self.logger.error(f"Error saving pair: {str(e)}")

    def process(self):
        self.scan_dataset()
        matched_data = self.match_images_labels()
        self.save_normalized_dataset(matched_data)
        print(f"Total images: {len(self.dataset_info['images'])}")
        print(f"Total labels: {len(self.dataset_info['labels'])}")
        print(f"Matched pairs: {len(matched_data)}")
        print(f"Unmatched images: {len(self.dataset_info['unmatched_images'])}")

if __name__ == "__main__":
    raw_dataset_path = "/UE/Master Thesis/raw_dataset/IAM historical"
    normalized_dataset_path = "/UE/Master Thesis/dataset_ready/IAM_historical"
    processor = IAMHistoricalProcessor(raw_dataset_path, normalized_dataset_path)
    processor.process()
