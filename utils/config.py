import torch
from torchvision import transforms as T

from PIL import Image
import os


class Config:
    # Paths
    train_val_images = "/home/etman/etman/python/projects/semantic_segmentation_project/pascal_vac_2012/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation"
    TRAIN_VAL_IMAGES = "/home/etman/etman/python/projects/semantic_segmentation_project/pascal_vac_2012/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/trainval.txt"
    TRAIN_IMAGES_PATHS = os.path.join(train_val_images, "train.txt")
    VAL_IMAGES_PATHS = os.path.join(train_val_images, "val.txt")
    ALL_IMAGES = "/home/etman/etman/python/projects/semantic_segmentation_project/pascal_vac_2012/VOC2012_train_val/VOC2012_train_val/JPEGImages"
    SEG_CLASSES = "/home/etman/etman/python/projects/semantic_segmentation_project/pascal_vac_2012/VOC2012_train_val/VOC2012_train_val/SegmentationClass"

    # Hyperparamters
    EPOCHS = 50
    LR = 0.0001
    BATCH_SIZE = 8

    # Some values
    IMG_SIZE = 256
    NUM_CHANNELS = 3
    NUM_CLASSES = 21

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    image_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    # T.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    ])
    mask_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE))
    ])
    TRANSFORMS_DICT = {
        "data": image_transforms,
        "mask": mask_transforms
    }