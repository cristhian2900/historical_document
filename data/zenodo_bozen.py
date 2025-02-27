import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import shutil
from tqdm import tqdm

class ZenodoBozenProcessor:
    def __init__(self, raw_dataset_path, normalized_dataset_path):
        self.raw_dataset_path = Path(raw_dataset_path)
        self.normalized_dataset_path = Path(normalized_dataset_path)
        self.image_extensions = ['.png', '.jpg', '.jpeg']
        self.label_extensions = ['.xml', '.page']
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

    def parse_xml_file(self, xml_file):
        labels = {}
        try:
            self.logger.info(f"Parsing XML file: {xml_file}")
            
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.logger.info(f"File content preview: {content[:200]}") 
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            self.logger.info(f"XML namespaces: {root.nsmap if hasattr(root, 'nsmap') else 'No namespaces'}")
            
            text_lines = root.findall(".//TextLine")
            if text_lines:
                text_content = []
                for line in text_lines:
                    unicode_elem = line.find(".//Unicode")
                    if unicode_elem is not None and unicode_elem.text:
                        text_content.append(unicode_elem.text.strip())
                
                if text_content:
                    image_id = xml_file.stem
                    labels[image_id] = " ".join(text_content)
                    self.logger.info(f"Strategy 1 found text for {image_id}")
            
            if not labels:
                text_regions = root.findall(".//TextRegion")
                if text_regions:
                    text_content = []
                    for region in text_regions:
                        for elem in region.findall(".//*"):
                            if elem.text and elem.text.strip():
                                text_content.append(elem.text.strip())
                    
                    if text_content:
                        image_id = xml_file.stem
                        labels[image_id] = " ".join(text_content)
                        self.logger.info(f"Strategy 2 found text for {image_id}")
            
            if not labels:
                text_content = []
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        text_content.append(elem.text.strip())
                
                if text_content:
                    image_id = xml_file.stem
                    labels[image_id] = " ".join(text_content)
                    self.logger.info(f"Strategy 3 found text for {image_id}")
            
            if labels:
                self.logger.info(f"Successfully extracted label for {xml_file.stem}")
                self.logger.info(f"Label preview: {next(iter(labels.values()))[:100]}...")
            else:
                self.logger.warning(f"No labels found in {xml_file}")
                
        except Exception as e:
            self.logger.error(f"Error parsing XML {xml_file}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return labels

    def scan_dataset(self):
        self.logger.info("Scanning dataset...")
        
        xml_files = []
        for root, _, files in os.walk(self.raw_dataset_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.label_extensions):
                    xml_files.append(Path(root) / file)
        
        self.logger.info(f"Found {len(xml_files)} XML files")
        
        for xml_file in tqdm(xml_files, desc="Parsing XML files"):
            labels = self.parse_xml_file(xml_file)
            self.dataset_info['labels'].update(labels)

        for root, _, files in os.walk(self.raw_dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    image_path = Path(root) / file
                    self.dataset_info['images'].append(image_path)
                    if len(self.dataset_info['images']) <= 5:
                        self.logger.info(f"Sample image path: {image_path}")

        self.logger.info(f"Found {len(self.dataset_info['images'])} images")
        self.logger.info(f"Found {len(self.dataset_info['labels'])} labels")

        sample_labels = list(self.dataset_info['labels'].items())[:5]
        self.logger.info("Sample labels:")
        for id_, text in sample_labels:
            self.logger.info(f"ID: {id_}, Text: {text[:100]}...")

    def extract_image_id(self, image_path):
        image_id = image_path.stem
        
        possible_ids = [
            image_id,
            image_id.split('_')[0],
            image_id.replace('img', '').strip()
        ]
        
        for id_candidate in possible_ids:
            if id_candidate in self.dataset_info['labels']:
                return id_candidate
        
        return image_id

    def match_images_labels(self):
        matched_data = []
        unmatched_images = set(self.dataset_info['images'])

        self.logger.info("Starting matching process...")
        
        for image_path in tqdm(self.dataset_info['images'], desc="Matching"):
            image_id = self.extract_image_id(image_path)
            self.logger.debug(f"Processing image: {image_path.name} -> ID: {image_id}")
            
            if image_id in self.dataset_info['labels']:
                matched_data.append({
                    'image_path': str(image_path),
                    'label': self.dataset_info['labels'][image_id]
                })
                unmatched_images.discard(image_path)
                self.logger.debug(f"Matched: {image_path.name}")
            else:
                self.logger.debug(f"No match found for: {image_path.name}")

        self.dataset_info['unmatched_images'] = list(unmatched_images)
        self.logger.info(f"Matching complete. Found {len(matched_data)} matches")
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
    raw_dataset_path = "/UE/Master Thesis/raw_dataset/zenodo dataset Bozen"
    normalized_dataset_path = "/UE/Master Thesis/dataset_ready/Zenodo"
    processor = ZenodoBozenProcessor(raw_dataset_path, normalized_dataset_path)
    processor.process()
