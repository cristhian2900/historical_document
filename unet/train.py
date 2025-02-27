import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
from tqdm import tqdm
from unet_dataset import UNetDataset
from unet import LightweightUNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    'data': {
        'images_dir': '/UE/Master Thesis/all_models_dataset/images',
        'labels_dir': '/UE/Master Thesis/all_models_dataset/labels',
        'batch_size': 16,
        'num_workers': 4,
        'pin_memory': True,
        'train_split': 0.8,
        'image_size': 64
    },
    'training': {
        'epochs': 10,
        'learning_rate': 0.001,
        'device': torch.device('mps'),
        'model_save_path': '/UE/Master Thesis/code/models/models_trained',
        'early_stopping_patience': 3,
        'scheduler_patience': 2
    }
}

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = 1 - (2. * (pred * target).sum() + 1e-5) / (pred.sum() + target.sum() + 1e-5)
        return 0.5 * bce_loss + 0.5 * dice_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_path):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), Path(save_path) / 'unet_best_model.h5')
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= CONFIG['training']['early_stopping_patience']:
            logger.info("Early stopping triggered")
            break

def main():

    Path(CONFIG['training']['model_save_path']).mkdir(parents=True, exist_ok=True)

    dataset = UNetDataset(
        CONFIG['data']['images_dir'],
        CONFIG['data']['labels_dir'],
        image_size=CONFIG['data']['image_size']
    )

    train_size = int(len(dataset) * CONFIG['data']['train_split'])
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['data']['batch_size'],
        shuffle=True,
        num_workers=CONFIG['data']['num_workers'],
        pin_memory=CONFIG['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['data']['batch_size'],
        shuffle=False,
        num_workers=CONFIG['data']['num_workers'],
        pin_memory=CONFIG['data']['pin_memory']
    )

    model = LightweightUNet().to(CONFIG['training']['device'])
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=CONFIG['training']['scheduler_patience'],
        factor=0.5
    )
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=CONFIG['training']['device'],
        num_epochs=CONFIG['training']['epochs'],
        save_path=CONFIG['training']['model_save_path']
    )

if __name__ == '__main__':
    main()