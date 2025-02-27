import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UNetDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, transform=None, image_size: int = 64):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size

        if transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485], std=[0.229])
            ])
        else:
            self.transform = transform

        self.valid_pairs = self._load_valid_pairs()
        logger.info(f"Successfully loaded {len(self.valid_pairs)} valid image-label pairs")

    def _load_valid_pairs(self) -> list:

        valid_pairs = []
        label_files = list(self.labels_dir.glob('*.txt'))
        logger.info(f"Found {len(label_files)} label files")
        
        for label_path in label_files:
            base_name = label_path.stem
            possible_img_names = [
                f"{base_name}.png",
                f"{base_name}.jpg",
                f"{base_name}.jpeg"
            ]
            
            for img_name in possible_img_names:
                img_path = self.images_dir / img_name
                if img_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        valid_pairs.append((str(img_path), str(label_path)))
                        break
                    except Exception as e:
                        logger.warning(f"Corrupted image {img_path}: {str(e)}")
        
        if not valid_pairs:
            raise ValueError(f"No valid pairs found in {self.images_dir} and {self.labels_dir}")
        
        return valid_pairs

    def _process_label_text(self, text: str) -> float:

        try:
            text = text.strip()
            if ' ' in text: 
                numbers = [float(num) for num in text.split() if num.isdigit()]
                return min(max(numbers) / 1000, 1.0) 
            else: 
                return min(float(text) / 100, 1.0)  
        except:
            return 0.0

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> dict:
        try:
            image_path, label_path = self.valid_pairs[idx]

            image = Image.open(image_path).convert('L') 
            if self.transform:
                image = self.transform(image)
            
            with open(label_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            intensity = self._process_label_text(text)
            mask = torch.full((1, self.image_size, self.image_size), intensity)
            
            return {
                'image': image,
                'mask': mask,
                'text': text
            }
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return self._get_default_item()

    def _get_default_item(self) -> dict:
        return {
            'image': torch.zeros((1, self.image_size, self.image_size)),
            'mask': torch.zeros((1, self.image_size, self.image_size)),
            'text': ''
        }