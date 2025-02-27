import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from pathlib import Path
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from generator import Generator
from discriminator import Discriminator

class SimpleImageDataset(Dataset):
    def __init__(self, images_path: Path, transform=None):
        self.images_path = Path(images_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                 std=[0.5, 0.5, 0.5])
        ])
        
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            for img_path in self.images_path.glob(f'*{ext}'):
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.image_files.append(img_path)
                except (IOError, UnidentifiedImageError):
                    print(f"Skipping corrupted image: {img_path}")
                    continue
        
        print(f"Found {len(self.image_files)} valid images")
        
        if len(self.image_files) > 100000:
            self.image_files = self.image_files[:100000]
            print(f"Limited dataset to 100,000 images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        while True:
            try:
                image_path = self.image_files[idx]
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image
            except (IOError, UnidentifiedImageError):
                print(f"Error loading image {image_path}, trying next image")
                idx = (idx + 1) % len(self.image_files)

def train_gan(config):
    device = torch.device(config['training']['device'])
    print(f"Using device: {device}")
    
    dataset = SimpleImageDataset(images_path=Path(config['dataset']['images_path']))
    
    if len(dataset) == 0:
        raise RuntimeError("No valid images found in the dataset")
    

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        drop_last=True  
    )
    

    generator = Generator(noise_dim=config['model']['noise_dim'], 
                        img_channels=config['model']['img_channels']).to(device)
    discriminator = Discriminator(img_channels=config['model']['img_channels']).to(device)
    

    optimizer_G = optim.Adam(generator.parameters(), lr=config['training']['lr'])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['training']['lr'])
    
  
    criterion = torch.nn.BCELoss()
    

    def real_label_smoothing():
        return torch.FloatTensor(config['training']['batch_size'], 1).uniform_(0.7, 1.0)
    
    def fake_label_smoothing():
        return torch.FloatTensor(config['training']['batch_size'], 1).uniform_(0.0, 0.3)


    save_dir = Path('saved_models')
    save_dir.mkdir(exist_ok=True)
    
    best_g_loss = float('inf')
    best_d_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        total_g_loss = 0
        total_d_loss = 0
        batches_processed = 0
        
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            
            real_labels = real_label_smoothing().to(device)
            fake_labels = fake_label_smoothing().to(device)
            
            optimizer_D.zero_grad()
            real_imgs = real_imgs.to(device)
            
            real_imgs = real_imgs + torch.randn_like(real_imgs) * 0.1
            
            d_loss_real = criterion(discriminator(real_imgs), real_labels)
            
            z = torch.randn(batch_size, config['model']['noise_dim']).to(device)
            fake_imgs = generator(z)
            d_loss_fake = criterion(discriminator(fake_imgs.detach()), fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            
           
            if d_loss.item() > 0.1:  
                optimizer_G.zero_grad()
                g_loss = criterion(discriminator(fake_imgs), real_labels)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            batches_processed += 1
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], '
                      f'Step [{i}/{len(dataloader)}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
        
        avg_d_loss = total_d_loss / batches_processed
        avg_g_loss = total_g_loss / batches_processed
        
        print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}] completed. '
              f'Average D Loss: {avg_d_loss:.4f}, Average G Loss: {avg_g_loss:.4f}')
        
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_path = save_dir / f'generator_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'loss': avg_g_loss,
            }, save_path)
            
            save_path = save_dir / f'discriminator_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_D.state_dict(),
                'loss': avg_d_loss,
            }, save_path)
            
            print(f'Saved models at epoch {epoch+1}')
        
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'loss': best_g_loss,
            }, save_dir / 'best_generator.pth')
        
        if avg_d_loss < best_d_loss:
            best_d_loss = avg_d_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_D.state_dict(),
                'loss': best_d_loss,
            }, save_dir / 'best_discriminator.pth')

CONFIG = {
    'model': {
        'noise_dim': 100,
        'img_channels': 3,
    },
    'training': {
        'epochs': 50, 
        'batch_size': 32,
        'lr': 0.0001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'save_interval': 5,  
    },
    'dataset': {
        'images_path': '/UE/Master Thesis/all_models_dataset/images',
    }
}

def main():
    try:
        train_gan(CONFIG)
    except Exception as e:
        print(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()