import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import optim
from torch.nn import CrossEntropyLoss

from baseline_one.unet.unet_model import UNet
from baseline_one.pretrained_model.resnetunet_model import ResNetUNet
from src.load_data import load_data
from src.training import train_and_val
from utils.config import Config
from utils.visualize import plot_losses

def main():
    train_loader, val_loader = load_data()
    # for images, masks in train_loader:
    #     print("images:", images.shape)
    #     print("masks:", masks.shape)
    #     break
    model = ResNetUNet(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    critirion = CrossEntropyLoss(ignore_index=255)

    model, train_losses, val_losses = train_and_val(model,
                                                    train_loader,
                                                    val_loader,
                                                    optimizer,
                                                    critirion,
                                                    Config.DEVICE,
                                                    Config.EPOCHS)
    
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()