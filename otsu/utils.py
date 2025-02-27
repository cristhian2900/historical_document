import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_images_from_folder(folder, max_images=100000):
    images = []
    skipped = 0
    folder_path = Path(folder)
    
    for img_path in folder_path.glob('*.*'):
        if len(images) >= max_images:
            break
            
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            continue
            
    print(f"Loaded {len(images)} images, skipped {skipped} files")
    return images

def split_dataset(images):
    train_size = int(0.8 * len(images))
    train_images = images[:train_size]
    val_images = images[train_size:]
    return train_images, val_images