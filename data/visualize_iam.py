import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image
import numpy as np
from collections import Counter
import pandas as pd

class IAMVisualizer:
    def __init__(self, dataset_path, visualization_path):
        self.dataset_path = Path(dataset_path)
        self.visualization_path = Path(visualization_path)
        self.visualization_path.mkdir(parents=True, exist_ok=True)
        
        self.matched_pairs_file = self.dataset_path / 'matched_pairs.json'
        self.unmatched_file = self.dataset_path / 'unmatched_images.json'
        
        with open(self.matched_pairs_file, 'r') as f:
            self.matched_pairs = json.load(f)
        with open(self.unmatched_file, 'r') as f:
            self.unmatched_images = json.load(f)
        
        self.valid_pairs = []
        self.validate_pairs()

    def validate_pairs(self):
        for pair in self.matched_pairs:
            try:
                image_path = Path(pair['image'])
                if image_path.exists():
                    with Image.open(image_path) as img:
                        self.valid_pairs.append(pair)
            except Exception:
                continue

    def plot_matching_statistics(self):
        total_images = len(self.matched_pairs) + len(self.unmatched_images)
        valid_count = len(self.valid_pairs)
        invalid_count = len(self.matched_pairs) - valid_count
        unmatched_count = len(self.unmatched_images)
        
        plt.figure(figsize=(10, 6))
        plt.bar(['Valid Matches', 'Invalid Images', 'Unmatched'], 
                [valid_count, invalid_count, unmatched_count],
                color=['green', 'yellow', 'red'])
        plt.title('IAM Dataset Matching Statistics')
        plt.ylabel('Number of Images')
        
        for i, v in enumerate([valid_count, invalid_count, unmatched_count]):
            percentage = (v / total_images) * 100
            plt.text(i, v, f'{v}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.savefig(self.visualization_path / 'iam_matching_statistics.png')
        plt.close()

    def visualize_random_samples(self, num_samples=1):
        if not self.valid_pairs:
            print("No valid pairs to visualize")
            return
        
        samples = random.sample(self.valid_pairs, min(num_samples, len(self.valid_pairs)))
        
        for idx, sample in enumerate(samples):
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                fig.suptitle(f'Sample {Path(sample["image"]).stem}', fontsize=12)
                
                image_path = Path(sample['image'])
                img = Image.open(image_path)
                ax1.imshow(img, cmap='gray' if img.mode == 'L' else None)
                ax1.set_title('Sample Image')
                ax1.axis('off')
                
                label_path = Path(sample['label'])
                with open(label_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                ax2.text(0.1, 0.5, f'Label Content:\n"{text}"', 
                        wrap=True, fontsize=10, 
                        verticalalignment='center')
                ax2.axis('off')
                
                plt.tight_layout()
                plt.savefig(self.visualization_path / f'iam_sample_{image_path.stem}.png')
                plt.close()
                
            except Exception as e:
                print(f"Error processing sample {idx+1}: {str(e)}")
                continue

    def analyze_image_dimensions(self):
        if not self.valid_pairs:
            print("No valid pairs to analyze")
            return
        
        dimensions = []
        for pair in self.valid_pairs:
            try:
                with Image.open(pair['image']) as img:
                    dimensions.append(img.size)
            except Exception:
                continue
        
        if not dimensions:
            print("No valid images found for dimension analysis")
            return
        
        widths, heights = zip(*dimensions)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(widths, heights, alpha=0.5)
        plt.title('IAM Dataset Image Dimensions')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.savefig(self.visualization_path / 'iam_dimensions.png')
        plt.close()
        
        return {
            'width': {'min': min(widths), 'max': max(widths), 'avg': sum(widths)/len(widths)},
            'height': {'min': min(heights), 'max': max(heights), 'avg': sum(heights)/len(heights)}
        }

    def generate_report(self):
        total_images = len(self.matched_pairs) + len(self.unmatched_images)
        valid_percentage = (len(self.valid_pairs) / total_images) * 100 if total_images > 0 else 0
        
        dimension_stats = self.analyze_image_dimensions()
        
        report = {
            'total_images': total_images,
            'valid_pairs': len(self.valid_pairs),
            'invalid_pairs': len(self.matched_pairs) - len(self.valid_pairs),
            'unmatched_images': len(self.unmatched_images),
            'valid_percentage': valid_percentage,
            'dimension_statistics': dimension_stats
        }
        
        with open(self.visualization_path / 'iam_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return report

    def visualize(self):
        print("\nGenerating IAM dataset visualizations...")
        self.plot_matching_statistics()
        self.visualize_random_samples()
        report = self.generate_report()
        
        print("\nIAM Dataset Statistics:")
        print(f"Total Images: {report['total_images']}")
        print(f"Valid Pairs: {report['valid_pairs']}")
        print(f"Invalid Pairs: {report['invalid_pairs']}")
        print(f"Unmatched Images: {report['unmatched_images']}")
        print(f"Valid Percentage: {report['valid_percentage']:.2f}%")
        
        if report.get('dimension_statistics'):
            width_stats = report['dimension_statistics']['width']
            height_stats = report['dimension_statistics']['height']
            print("\nImage Dimensions:")
            print(f"Width - Min: {width_stats['min']}, Max: {width_stats['max']}, Avg: {width_stats['avg']:.2f}")
            print(f"Height - Min: {height_stats['min']}, Max: {height_stats['max']}, Avg: {height_stats['avg']:.2f}")
        
        print(f"\nVisualizations saved to: {self.visualization_path}")

if __name__ == "__main__":
    dataset_path = "/UE/Master Thesis/dataset_ready/IAM_dataset"
    visualization_path = "/UE/Master Thesis/visualize"
    
    visualizer = IAMVisualizer(dataset_path, visualization_path)
    visualizer.visualize()
