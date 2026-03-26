import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import os

class SegmentationDataset(Dataset):
    def __init__(self, all_images, seg_classes, images_path_file, transforms=None):
        self.all_images = all_images
        self.seg_classes = seg_classes
        self.images_path_file = images_path_file
        self.transforms = transforms

        self.images_paths = []
        self.images_masks = []

        with open(self.images_path_file, 'r') as f:
            data = f.readlines()
        for img in data:
            img_name = img.strip() 
            img_path = os.path.join(self.all_images, img_name + ".jpg")
            img_mask_path = os.path.join(self.seg_classes, img_name + ".png")
            self.images_paths.append(img_path)
            self.images_masks.append(img_mask_path)

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        img = self.images_paths[idx]
        mask = self.images_masks[idx]

        img = Image.open(img).convert("RGB")
        mask = Image.open(mask)

        if self.transforms:
            img = self.transforms["data"](img)
            mask = self.transforms["mask"](mask)

        return img, torch.tensor(np.array(mask), dtype=torch.long)