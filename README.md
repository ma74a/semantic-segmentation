<div align="center">

# 🧠 Semantic Segmentation — Pascal VOC 2012

**A PyTorch deep learning project for pixel-wise semantic segmentation**  
*Three architectures · Combined loss · Pascal VOC 2012 · 21 Classes*

</div>

---

## 📖 Overview

This project implements and compares **three semantic segmentation architectures** on the Pascal VOC 2012 dataset. Semantic segmentation assigns a class label to every pixel in an image — enabling applications like autonomous driving, medical imaging, and scene understanding.

The project features:
- ✅ Three model baselines ready to train and compare
- ✅ Combined **CrossEntropy + Dice** loss for robust training
- ✅ Proper `ignore_index=255` handling for VOC boundary pixels
- ✅ Automatic best-model checkpointing
- ✅ Loss curve visualization saved to `loss_plot.png`
- ✅ Single-image inference pipeline
- ✅ **Streamlit web app** for interactive image segmentation

---

## 🏗️ Model Architectures

### 1. UNet (Scratch)
> `baseline_one/unet/`

A classic **encoder–decoder** architecture with skip connections. Trained entirely from scratch.

```
Input (3, H, W)
    │
    ▼
[DoubleConv] ──────────────────────────────────────────────► skip1 (64)
    │
[Down → DoubleConv] ───────────────────────────────────────► skip2 (128)
    │
[Down → DoubleConv] ───────────────────────────────────────► skip3 (256)
    │
[Down → DoubleConv] ───────────────────────────────────────► skip4 (512)
    │
[Bottleneck: Down → DoubleConv] (1024)
    │
[Up + skip4 → DoubleConv] (512)
    │
[Up + skip3 → DoubleConv] (256)
    │
[Up + skip2 → DoubleConv] (128)
    │
[Up + skip1 → DoubleConv] (64)
    │
[1×1 Conv]
    │
    ▼
Output (num_classes, H, W)
```

---

### 2. ResNetUNet (Pretrained Encoder)
> `baseline_one/pretrained_model/`

A hybrid model using a **pretrained ResNet34 encoder** paired with a custom UNet-style decoder. Benefits from ImageNet pretraining for richer feature extraction.

| Stage | Layer | Output Channels |
|---|---|---|
| Encoder 0 | Conv1 → BN → ReLU | 64 |
| Pool | MaxPool | — |
| Encoder 1 | ResNet Layer1 | 64 |
| Encoder 2 | ResNet Layer2 | 128 |
| Encoder 3 | ResNet Layer3 | 256 |
| Encoder 4 (Bottleneck) | ResNet Layer4 | 512 |
| Decoder 4 | Upsample + skip(e3) + DoubleConv | 256 |
| Decoder 3 | Upsample + skip(e2) + DoubleConv | 128 |
| Decoder 2 | Upsample + skip(e1) + DoubleConv | 64 |
| Decoder 1 | Upsample + skip(e0) + DoubleConv | 64 |
| Decoder 0 | Upsample + DoubleConv | 64 |
| Output | 1×1 Conv | num_classes |

---

### 3. DeepLabV3 with ResNet50 (Fine-tuned) ← *Currently Active*
> `baseline_two/resnet_deeplab.py`

The strongest baseline — a **pretrained DeepLabV3** model with ResNet50 backbone loaded from `torchvision`. Only the final classification head is replaced and fine-tuned for 21 VOC classes.

```python
# Only this layer is trained from scratch:
in_channels = model.classifier[4].in_channels
model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
```

---

## 📁 Project Structure

```
semantic-segmentation/
│
├── 📂 baseline_one/
│   ├── 📂 unet/
│   │   ├── unet_model.py          # UNet full model
│   │   └── unet_parts.py          # DoubleConv, Down, Up blocks
│   └── 📂 pretrained_model/
│       └── resnetunet_model.py    # ResNet34 + UNet decoder
│
├── 📂 baseline_two/
│   └── resnet_deeplab.py          # DeepLabV3 fine-tuning wrapper
│
├── 📂 src/
│   ├── dataset.py                 # SegmentationDataset (PyTorch Dataset)
│   ├── load_data.py               # DataLoader factory (80/20 random split)
│   └── training.py                # Training & validation loop + checkpointing
│
├── 📂 utils/
│   ├── config.py                  # Central config (paths, hyperparameters, transforms)
│   ├── helper.py                  # DiceLoss implementation
│   └── visualize.py               # Loss curves & batch visualization
│
├── 📂 app/
│   └── app.py                     # 🌐 Streamlit GUI for interactive inference
│
├── 📂 testing/
│   └── 2007_000256.jpg            # Sample test image
│
├── 📂 checkpoints/                # Saved model weights (auto-created)
├── 📂 pascal_vac_2012/            # Dataset (not tracked in git)
│
├── train_model.py                 # 🚀 Training entry point
├── predict.py                     # 🔮 Inference on a single image
└── requirements.txt
```

---

## 📦 Dataset Setup

This project uses the **Pascal VOC 2012** segmentation benchmark.

**Download:** [Official VOC website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) · [Kaggle mirror](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)

After downloading, organize the data to match this structure:

```
pascal_vac_2012/
└── VOC2012_train_val/
    └── VOC2012_train_val/
        ├── JPEGImages/            ← RGB input images   (.jpg)
        ├── SegmentationClass/     ← Pixel-wise masks    (.png)
        └── ImageSets/
            └── Segmentation/
                ├── train.txt      ← 1,464 training IDs
                ├── val.txt        ← 1,449 validation IDs
                └── trainval.txt   ← Combined 2,913 IDs  ← used by default
```

> **Boundary pixels** with label `255` are automatically excluded from training via `ignore_index=255`.

---

## ⚙️ Configuration

All settings live in [`utils/config.py`](utils/config.py). The actual current values are:

```python
class Config:
    # ── Paths (update these to match your system) ──────────────────────
    TRAIN_VAL_IMAGES   = ".../ImageSets/Segmentation/trainval.txt"  # used by load_data
    TRAIN_IMAGES_PATHS = ".../ImageSets/Segmentation/train.txt"     # available but unused
    VAL_IMAGES_PATHS   = ".../ImageSets/Segmentation/val.txt"       # available but unused
    ALL_IMAGES         = ".../JPEGImages"
    SEG_CLASSES        = ".../SegmentationClass"

    # ── Hyperparameters ────────────────────────────────────────────────
    EPOCHS      = 50
    LR          = 0.0001     # Adam optimizer
    BATCH_SIZE  = 8

    # ── Model Settings ─────────────────────────────────────────────────
    IMG_SIZE    = 256        # Images & masks resized to 256×256
    NUM_CHANNELS = 3
    NUM_CLASSES = 21         # Background + 20 VOC categories

    # ── Device ─────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 🚀 Installation

**Requirements:** Python 3.10+, pip, and a CUDA-capable GPU (recommended)

```bash
# 1. Clone the repository
git clone <repo-url>
cd semantic-segmentation

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

<details>
<summary><b>📋 Full dependency list</b></summary>

| Package | Min Version | Purpose |
|---|---|---|
| `torch` | 2.0.0 | Deep learning framework |
| `torchvision` | 0.15.0 | Pretrained models & transforms |
| `Pillow` | 9.0.0 | Image I/O |
| `numpy` | 1.23.0 | Array operations |
| `matplotlib` | 3.5.0 | Plotting loss curves |
| `tqdm` | 4.64.0 | Training progress bars |
| `streamlit` | 1.0.0 | Interactive web app UI |

</details>

---

## 🏋️ Training

### Step 1 — Update paths in `utils/config.py`

Set `ALL_IMAGES`, `SEG_CLASSES`, and `TRAIN_VAL_IMAGES` to your local dataset paths.

### Step 2 — Choose a model

Open [`train_model.py`](train_model.py) and select one of the three architectures:

```python
# Option A — UNet from scratch
model = UNet(in_channels=Config.NUM_CHANNELS, num_classes=Config.NUM_CLASSES)

# Option B — ResNet34 encoder + UNet decoder (pretrained)
model = ResNetUNet(num_classes=Config.NUM_CLASSES)

# Option C — DeepLabV3 ResNet50 fine-tuned (best results) ← currently active
model = ResnetDeeplab(num_classes=Config.NUM_CLASSES)
```

### Step 3 — Run training

```bash
python train_model.py
```

**What happens during training:**
1. `trainval.txt` is loaded and randomly split (80% train / 20% val)
2. Each epoch computes `CE Loss + Dice Loss` on both splits
3. If val loss improves → checkpoint saved to `checkpoints/model1.pth` as a full state dict `{model_state_dict, optimizer_state_dict, epoch}`
4. After all epochs → `loss_plot.png` is saved to the project root

### Training Output Example

```
1464 365
epoch: 1  | train_loss: 2.4312 | val_loss: 2.1874
epoch: 2  | train_loss: 1.9801 | val_loss: 1.7653
...
```

---

## 🔮 Inference

Run inference on a single image:

```bash
python predict.py
```

The script loads `checkpoints/resnetunet_model_v1.pt` using `ResNetUNet` and displays a side-by-side comparison:

```
┌─────────────────┬─────────────────────┐
│   Input Image   │   Predicted Mask    │
│  (RGB photo)    │  (class color map)  │
└─────────────────┴─────────────────────┘
```

To test on your own image, change the path in `predict.py`:

```python
img_path = "./testing/your_image.jpg"
```

> ⚠️ Make sure the model class in `predict.py` matches the checkpoint you trained. Currently `predict.py` loads `ResNetUNet` while `train_model.py` trains `ResnetDeeplab`.

---

## 🌐 Streamlit App

An interactive web UI is available for running inference without writing any code.

```bash
streamlit run app/app.py
```

The app will open in your browser and lets you:
- **Upload** a `.jpg` or `.png` image
- **View** the original image
- **See** the predicted segmentation mask side-by-side

The app uses the `ResNetUNet` model and loads weights from `checkpoints/resnetunet_model_v1.pt`.

> ⚠️ Make sure `streamlit` is installed (`pip install streamlit`) and that the checkpoint path inside `app/app.py` points to your trained weights.

---

## 📉 Loss Function

Training minimizes a **combined loss** for pixel-accurate and shape-aware predictions:

$$\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{Dice}$$

| Loss | Role |
|---|---|
| **CrossEntropyLoss** | Per-pixel classification; handles class probabilities |
| **DiceLoss** | Overlap-based loss; robust to class imbalance |

Both losses exclude boundary pixels (`label == 255`) via `ignore_index=255`.

---

## 🗺️ Pascal VOC 2012 — Class Labels

<table>
<tr>
  <th>ID</th><th>Class</th>
  <th>ID</th><th>Class</th>
  <th>ID</th><th>Class</th>
</tr>
<tr>
  <td>0</td><td>🟫 Background</td>
  <td>8</td><td>🐱 Cat</td>
  <td>16</td><td>🪴 Potted Plant</td>
</tr>
<tr>
  <td>1</td><td>✈️ Aeroplane</td>
  <td>9</td><td>🪑 Chair</td>
  <td>17</td><td>🐑 Sheep</td>
</tr>
<tr>
  <td>2</td><td>🚲 Bicycle</td>
  <td>10</td><td>🐄 Cow</td>
  <td>18</td><td>🛋️ Sofa</td>
</tr>
<tr>
  <td>3</td><td>🐦 Bird</td>
  <td>11</td><td>🍽️ Dining Table</td>
  <td>19</td><td>🚂 Train</td>
</tr>
<tr>
  <td>4</td><td>⛵ Boat</td>
  <td>12</td><td>🐶 Dog</td>
  <td>20</td><td>📺 TV/Monitor</td>
</tr>
<tr>
  <td>5</td><td>🍾 Bottle</td>
  <td>13</td><td>🐴 Horse</td>
  <td>255</td><td>⬜ Boundary (ignored)</td>
</tr>
<tr>
  <td>6</td><td>🚌 Bus</td>
  <td>14</td><td>🏍️ Motorbike</td>
  <td></td><td></td>
</tr>
<tr>
  <td>7</td><td>🚗 Car</td>
  <td>15</td><td>🧍 Person</td>
  <td></td><td></td>
</tr>
</table>

---

## 📄 License

This project is for educational and research purposes.  
The Pascal VOC 2012 dataset is subject to its own [terms of use](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

---

<div align="center">

Made with ❤️ using PyTorch

</div>
