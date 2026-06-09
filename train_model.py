import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import optim
from torch.nn import CrossEntropyLoss

from baseline_one.pretrained_model.resnetunet_model import ResNetUNet
from baseline_two.resnet_deeplab import ResnetDeeplab
from baseline_one.unet.unet_model import UNet
from utils.visualize import plot_losses
from src.training import train_and_val
from src.load_data import load_data
from utils.config import Config
from utils.helper import DiceLoss

def main():
    train_loader, val_loader = load_data()
    # for images, masks in train_loader:
    #     print("images:", images.shape)
    #     print("masks:", masks.shape)
    #     break
    model = ResnetDeeplab(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    cl_loss = CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(num_classes=Config.NUM_CLASSES)

    model, train_losses, val_losses = train_and_val(model,
                                                    train_loader,
                                                    val_loader,
                                                    optimizer,
                                                    cl_loss,
                                                    dice_loss,
                                                    Config.DEVICE,
                                                    Config.EPOCHS)
    
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()