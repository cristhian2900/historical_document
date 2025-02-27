import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time
from tqdm import tqdm
import logging
from rcnn_dataset import RCNNDataset
from model import RCNN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RCNNTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = config['training']['device']
        self.model = self._initialize_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=1e-5
        )

    def _initialize_model(self) -> RCNN:
        model = RCNN(num_classes=3).to(self.device)
        return model

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        best_val_loss = float('inf')
        save_path = Path(self.config['training']['model_save_path'])
        save_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config['training']['epochs']):
            train_loss = self._train_epoch(train_loader, epoch)
            
            val_loss = self._validate(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = save_path / 'rcnn_best_model.h5'
                self.model.save_model(str(model_path))
                logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["training"]["epochs"]}')
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs, _ = self.model(images, boxes)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        return total_loss / len(train_loader)

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    boxes = batch['boxes'].to(self.device)
                    
                    outputs, _ = self.model(images, boxes)
                    loss = self.criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        return total_loss / len(val_loader)

def main():
    try:
        config = {
            'data': {
                'images_dir': '/UE/Master Thesis/all_models_dataset/images',
                'labels_dir': '/UE/Master Thesis/all_models_dataset/labels',
                'batch_size': 8
            },
            'training': {
                'epochs': 15,
                'learning_rate': 1e-4,
                'device': torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
                'model_save_path': '/UE/Master Thesis/code/models/models_trained'
            }
        }
        
        dataset = RCNNDataset(config['data']['images_dir'], config['data']['labels_dir'])
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=RCNNDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=RCNNDataset.collate_fn
        )
        
        trainer = RCNNTrainer(config)
        trainer.train(train_loader, val_loader)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
