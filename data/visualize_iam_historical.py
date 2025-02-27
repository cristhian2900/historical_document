import matplotlib.pyplot as plt
import random
from pathlib import Path
import logging
from datetime import datetime
import os

class IAMHistoricalVisualizer:
    def __init__(self, normalized_dataset_path, visualization_path):
        self.normalized_dataset_path = Path(normalized_dataset_path)
        self.visualization_path = Path(visualization_path)
        self.images_path = self.normalized_dataset_path / 'images'
        self.labels_path = self.normalized_dataset_path / 'labels'
        
        self.visualization_path.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        log_file = self.visualization_path / f'visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def load_image_label_pairs(self):

        pairs = []
        try:
            image_files = list(self.images_path.glob('*.png')) 
            self.logger.info(f"Found {len(image_files)} images")

            for image_path in image_files:
                label_path = self.labels_path / f"{image_path.stem}.txt"
                if label_path.exists():
                    try:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            label_content = f.read().strip()
                            pairs.append({
                                'image_path': image_path,
                                'label': label_content
                            })
                    except Exception as e:
                        self.logger.error(f"Error reading label for {image_path.name}: {str(e)}")

            self.logger.info(f"Successfully loaded {len(pairs)} image-label pairs")
            return pairs
        except Exception as e:
            self.logger.error(f"Error loading pairs: {str(e)}")
            return []

    def create_sample_visualization(self, num_samples=4, save=True, show=True):
        all_pairs = self.load_image_label_pairs()
        
        if not all_pairs:
            self.logger.warning("No image-label pairs found to visualize.")
            return

        selected_pairs = random.sample(all_pairs, min(num_samples, len(all_pairs)))
        
        fig = plt.figure(figsize=(15, 5 * num_samples))
        
        for idx, pair in enumerate(selected_pairs):
            try:
                img = plt.imread(str(pair['image_path']))
                ax_img = plt.subplot(num_samples, 2, 2 * idx + 1)
                ax_img.imshow(img, cmap='gray')
                ax_img.set_title(f"Sample {pair['image_path'].stem}")
                ax_img.axis('off')
                
                ax_text = plt.subplot(num_samples, 2, 2 * idx + 2)
                ax_text.text(0.5, 0.5, f"Label: {pair['label']}", 
                           fontsize=12, ha='center', va='center', 
                           wrap=True)
                ax_text.axis('off')
                
            except Exception as e:
                self.logger.error(f"Error processing pair {idx}: {str(e)}")
                continue
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.visualization_path / f'iam_historical_samples_{timestamp}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            self.logger.info(f"Visualization saved to: {save_path}")
            print(f"Visualization saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

if __name__ == "__main__":
    normalized_dataset_path = "/UE/Master Thesis/dataset_ready/IAM_historical"
    visualization_path = "/UE/Master Thesis/code/visualization"
    
    visualizer = IAMHistoricalVisualizer(normalized_dataset_path, visualization_path)
    visualizer.create_sample_visualization(num_samples=4, save=True, show=True)