import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np

from baseline_one.pretrained_model.resnetunet_model import ResNetUNet
from utils.config import Config


# load the model
@st.cache_resource
def load_model():
    model = ResNetUNet(num_classes=Config.NUM_CLASSES)
    checkpoints = torch.load("/home/etman/etman/projects/semantic-segmentation/checkpoints/resnetunet_model_v1.pt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoints)
    model.eval()
    return model


model = load_model()

# UI
st.title("Semantic Segmentation App")
st.write("Upload an image and see predicted mask")

uploaded_file = st.file_uploader("choose an image", type=["jpg", "png"])

if uploaded_file is not None:
    # load the image
    img = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original Image")
    st.image(img, use_container_width=True)

    # preprocessing the image
    input_img = Config.TRANSFORMS_DICT["data"](img)
    input_img = input_img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(input_img)
        pred = torch.argmax(output, dim=1)[0].cpu().numpy()

    
    st.subheader("Predicted Mask")
    fig, ax = plt.subplots()
    ax.imshow(pred)
    ax.axis("off")
    st.pyplot(fig)