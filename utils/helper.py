import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W]
        # targets: [B, H, W]
        probs = F.softmax(logits, dim=1)

        valid_mask = (targets != self.ignore_index)

        targets = targets.clone()
        targets[~valid_mask] = 0

        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1)

        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()