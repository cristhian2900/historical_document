import os
import json
from pathlib import Path
import logging
import shutil
from tqdm import tqdm

class BenthamDatasetProcessor:
    def __init__(self, raw_dataset_path, normalized_dataset_path):
        self.raw_dataset_path = Path(raw_dataset_path)
        self.normalized_dataset_path = Path(normalized_dataset_path)
        self.image_extensions = ['.png', '.jpg', '.jpeg']
        self.label_extensions = ['.txt', '.json', '.xml']
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

    def find_label_files(self):

        label_files = []
        
        
        possible_label_paths = [
            self.raw_dataset_path / "Transcriptions",
            self.raw_dataset_path / "Labels",
            self.raw_dataset_path / "ground_truth",
            self.raw_dataset_path
        ]

        for path in possible_label_paths:
            if path.exists():
                self.logger.info(f"Searching for labels in: {path}")
                for ext in self.label_extensions:
                    label_files.extend(path.rglob(f"*{ext}"))

        self.logger.info(f"Found {len(label_files)} label files")
        return label_files

    def parse_label_file(self, label_file):
      
        labels = {}
        try:
            with open(label_file, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                
                if label_file.suffix == '.json':
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            labels = data
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'id' in item and 'text' in item:
                                    labels[item['id']] = item['text']
                    except json.JSONDecodeError:
                        pass

                elif label_file.suffix in ['.txt', '.xml']:

                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                         
                            if len(lines) == 1:
                                image_id = label_file.stem
                                labels[image_id] = line
                            else:
                             
                                parts = line.split(maxsplit=1)
                                if len(parts) == 2:
                                    labels[parts[0]] = parts[1]
                                elif len(parts) == 1:
                                
                                    labels[parts[0]] = parts[0]

            if labels:
                self.logger.info(f"Extracted {len(labels)} labels from {label_file}")
            
                sample_labels = list(labels.items())[:2]
                self.logger.info(f"Sample labels from {label_file}: {sample_labels}")
            
        except Exception as e:
            self.logger.error(f"Error parsing {label_file}: {str(e)}")
        
        return labels

    def scan_dataset(self):
        self.logger.info("Scanning dataset...")
        
        label_files = self.find_label_files()
        for label_file in tqdm(label_files, desc="Parsing label files"):
            labels = self.parse_label_file(label_file)
            self.dataset_info['labels'].update(labels)

       
        for root, _, files in os.walk(self.raw_dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    image_path = Path(root) / file
                    self.dataset_info['images'].append(image_path)

        self.logger.info(f"Found {len(self.dataset_info['images'])} images")
        self.logger.info(f"Found {len(self.dataset_info['labels'])} labels")

    def extract_image_id(self, image_path):

        possible_ids = [
            image_path.stem,  
            image_path.stem.split('_')[0],  
            image_path.stem.split('-')[0],  
        ]
        
        for id_candidate in possible_ids:
            if id_candidate in self.dataset_info['labels']:
                return id_candidate
        
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
    raw_dataset_path = "/UE/Master Thesis/raw_dataset/Bentham Dataset"
    normalized_dataset_path = "/UE/Master Thesis/dataset_ready/Bentham"
    processor = BenthamDatasetProcessor(raw_dataset_path, normalized_dataset_path)
    processor.process()
