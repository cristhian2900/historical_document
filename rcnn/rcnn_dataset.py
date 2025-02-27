import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCNNDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.image_files = list(self.images_dir.glob('*.png'))
        
        self.valid_pairs = []
        for img_path in self.image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.valid_pairs.append((img_path, label_path))
                    logger.debug(f"Valid pair found: {img_path.name} - {label_path.name}")
                except Exception as e:
                    logger.warning(f"Skipping corrupted image {img_path}: {str(e)}")
            else:
                logger.warning(f"No label found for image: {img_path.name}")

        if not self.valid_pairs:
            raise ValueError("No valid image-label pairs found.")
        
        logger.info(f"Found {len(self.valid_pairs)} valid image-label pairs")

        self.label_categories = {
            'numeric': 0,
            'sentence': 1,
            'word': 2
        }

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Dict:

        max_retries = 3
        current_idx = idx
        
        for _ in range(max_retries):
            try:
                image_path, label_path = self.valid_pairs[current_idx]

                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                with open(label_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                if any(c.isdigit() for c in text):
                    label = self.label_categories['numeric']
                elif len(text.split()) > 3:
                    label = self.label_categories['sentence']
                else:
                    label = self.label_categories['word']

                boxes = torch.tensor([[0, 0, 223, 223]], dtype=torch.float32)
                
                return {
                    'image': image,
                    'label': torch.tensor(label, dtype=torch.long),
                    'boxes': boxes,
                    'text': text
                }
                
            except (UnidentifiedImageError, OSError) as e:
                logger.error(f"Error loading image {image_path}: {e}")

                current_idx = (current_idx + 1) % len(self.valid_pairs)
            except Exception as e:
                logger.error(f"Unexpected error processing item {current_idx}: {e}")
                current_idx = (current_idx + 1) % len(self.valid_pairs)
        
        logger.error(f"Failed to load item after {max_retries} retries, returning default item")
        return self._get_default_item()

    def _get_default_item(self) -> Dict:

        return {
            'image': torch.zeros((3, 224, 224)),
            'label': torch.tensor(0, dtype=torch.long),
            'boxes': torch.tensor([[0, 0, 223, 223]], dtype=torch.float32),
            'text': ''
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        images = []
        labels = []
        boxes = []
        texts = []
        
        for b in batch:
            images.append(b['image'])
            labels.append(b['label'])
            boxes.append(b['boxes'])
            texts.append(b['text'])
        
        return {
            'image': torch.stack(images),
            'label': torch.stack(labels),
            'boxes': torch.stack(boxes),
            'text': texts
        }
