import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from .dataset import SegmentationDataset
from utils.config import Config

def load_data():
    train_dataset = SegmentationDataset(
        all_images=Config.ALL_IMAGES,
        seg_classes=Config.SEG_CLASSES,
        images_path_file=Config.TRAIN_IMAGES_PATHS,
        transforms=Config.TRANSFORMS_DICT
    )
    val_dataset = SegmentationDataset(
        all_images=Config.ALL_IMAGES,
        seg_classes=Config.SEG_CLASSES,
        images_path_file=Config.VAL_IMAGES_PATHS,
        transforms=Config.TRANSFORMS_DICT
    )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=Config.BATCH_SIZE)

    return train_loader, val_loader