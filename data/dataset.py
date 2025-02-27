import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import logging

class DocumentDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        self.setup_logging()
        
        self.prepared_dataset_path = Path("/UE/Master Thesis/dataset_ready/prepared_dataset")
        
        self.images = list(self.prepared_dataset_path.glob('*.png')) + \
                     list(self.prepared_dataset_path.glob('*.jpg')) + \
                     list(self.prepared_dataset_path.glob('*.jpeg'))
        
        self.labels = []
        for img_path in self.images:
            label_path = self.prepared_dataset_path / f"{img_path.stem}.txt"
            if label_path.exists():
                self.labels.append(label_path)
            else:
                self.logger.warning(f"No label found for {img_path}")
        
        self.images = [img for img, lbl in zip(self.images, self.labels) if lbl.exists()]
        
        total_size = len(self.images)
        split_idx = int(0.8 * total_size)
        
        if split == 'train':
            self.images = self.images[:split_idx]
            self.labels = self.labels[:split_idx]
        else:  
            self.images = self.images[split_idx:]
            self.labels = self.labels[split_idx:]
        
        self.logger.info(f"Loaded {len(self.images)} images for {split} split")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  
        
        label_path = self.labels[idx]
        with open(label_path, 'r') as f:
            label = int(f.read().strip())  
        
        if self.transform:
            image = self.transform(image)
        
        return image, label