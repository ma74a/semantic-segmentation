import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class ResnetDeeplab(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(ResnetDeeplab, self).__init__()
        self.model = deeplabv3_resnet50(weights="DEFAULT")

        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.model(x)
        return out["out"]