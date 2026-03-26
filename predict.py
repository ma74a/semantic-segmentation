import torch

from PIL import Image
import matplotlib.pyplot as plt

from baseline_one.unet.unet_model import UNet
from utils.config import Config


model = UNet(in_channels=Config.NUM_CHANNELS, num_classes=Config.NUM_CLASSES)
checkpoint = torch.load("./checkpoints/model1.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

img_path = "./test/2007_000256.jpg"
img = Image.open(img_path).convert("RGB")

img = Config.TRANSFORMS_DICT["data"](img)
img = img.unsqueeze(0)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1)

plt.figure(figsize=(12,5))

# original image
img = img.squeeze(0)
plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(img.permute(1, 2, 0))

# predicted mask
plt.subplot(1,2,2)
plt.title("Predicted Mask")
plt.imshow(pred[0])
plt.show()
