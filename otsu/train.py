import os
import cv2
import numpy as np
from threshold import otsu_thresholding
from utils import load_images_from_folder, split_dataset
import torch
from pathlib import Path
from tqdm import tqdm

def process_batch(images, batch_size=32):
    thresholded_images = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = []
        for img in batch:
            try:
                thresholded = otsu_thresholding(img)
                batch_results.append(thresholded)
            except Exception as e:
                continue
        thresholded_images.extend(batch_results)
    return thresholded_images

def main():
    dataset_path = Path('/UE/Master Thesis/all_models_dataset/images')
    model_save_path = Path('/UE/Master Thesis/code/models/models_trained')
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading images...")
    images = load_images_from_folder(dataset_path)
    
    print("Splitting dataset...")
    train_images, val_images = split_dataset(images)
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    print("Processing training images...")
    processed_train = process_batch(train_images)
    
    print("Processing validation images...")
    processed_val = process_batch(val_images)
    
    model_params = {
        'threshold_values': [],
        'image_size': (64, 64),
        'num_train_images': len(processed_train),
        'num_val_images': len(processed_val)
    }
    
    for img in processed_train:
        _, threshold_used = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        model_params['threshold_values'].append(threshold_used)
    
    model_params['avg_threshold'] = np.mean(model_params['threshold_values'])
    
    torch.save(model_params, model_save_path / 'otsu_model.pth')
    print(f"Model saved to {model_save_path / 'otsu_model.pth'}")
    
    print("Average threshold value:", model_params['avg_threshold'])
    print("Processing completed successfully")

if __name__ == "__main__":
    main()
