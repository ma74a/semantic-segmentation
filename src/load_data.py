import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, random_split
from typing import Tuple

from .dataset import SegmentationDataset
from utils.config import Config

def load_data() -> Tuple[DataLoader, DataLoader]:
    # train_dataset = SegmentationDataset(
    #     all_images=Config.ALL_IMAGES,
    #     seg_classes=Config.SEG_CLASSES,
    #     images_path_file=Config.TRAIN_IMAGES_PATHS,
    #     transforms=Config.TRANSFORMS_DICT
    # )
    # val_dataset = SegmentationDataset(
    #     all_images=Config.ALL_IMAGES,
    #     seg_classes=Config.SEG_CLASSES,
    #     images_path_file=Config.VAL_IMAGES_PATHS,
    #     transforms=Config.TRANSFORMS_DICT
    # )
    dataset = SegmentationDataset(Config.ALL_IMAGES, 
                                  seg_classes=Config.SEG_CLASSES,
                                  images_path_file=Config.TRAIN_VAL_IMAGES,
                                  transforms=Config.TRANSFORMS_DICT)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=Config.BATCH_SIZE)

    return train_loader, val_loader

