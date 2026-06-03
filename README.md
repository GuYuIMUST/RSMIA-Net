# DDAG-Net: Dual-Domain Adaptive Decoupling and Artery-Guided Network for Pulmonary Embolism Segmentation

> A novel 3D segmentation framework combining Transformer, quadruplet attention, and DA attention for pulmonary embolism detection in CTPA images.

---

## 📌 Overview

DDAG-Net is a 3D medical image segmentation framework designed for **pulmonary embolism (PE) segmentation** in **CT pulmonary angiography (CTPA)**. The network integrates multiple advanced attention mechanisms and a Transformer architecture to address the key challenges in PE analysis:

- **Small embolism targets** in pulmonary arteries
- **Complex 3D vascular structures**
- **Severe class imbalance** between emboli and background
- **Weak boundary representation** of emboli
- **Long-range dependency** modeling in 3D space

---

## 🧠 Network Architecture

The architecture is built upon an improved 3D U-Net with the following key components:

### Encoder Path

- **Input**: 3D CTPA volumes (single channel)
- **Stem**: Initial convolution and depthwise convolution
- **Four encoding stages** with progressive channel expansion (64→96→128→160)
- Each stage includes:
  - Basic convolutional blocks
  - Downsampling operations
  - DA attention modules (stages 3 and 4)

### Bottleneck

- **Transformer module**: 6-layer encoder-decoder with 8-head attention
- **3D positional encoding** for spatial awareness
- **ASPP module** for multi-scale feature extraction

### Decoder Path

- Progressive upsampling with skip connections
- **Quadruplet attention** at each decoding stage
- **DA attention** for enhanced feature representation
- Multi-level feature fusion

### Output Heads

- **Vessel segmentation head**: Auxiliary task for vascular structure
- **PE segmentation head**: Main task for embolism detection

---

## ✨ Key Innovations

### 1️⃣ Transformer with 3D Positional Encoding

Custom 3D positional encoding integrated with a standard Transformer architecture:

- Captures long-range spatial dependencies
- Preserves 3D structural information
- Enables global context understanding

```python
self.transformer = Transformer(
    d_model=128, nhead=8,
    num_encoder_layers=6, num_decoder_layers=6,
    dim_feedforward=2048, dropout=0.1
)
### 2️⃣ Quadruplet Attention Mechanism

Four attention modules strategically placed in the decoder:

- **Attention 1-3**: Applied to skip connections
- **Attention 4**: Applied to multi-scale feature fusion

Enhances feature discrimination in critical regions.

### 3️⃣ DA Attention (Dual Attention Network)

Integrated at multiple scales:

- **DA3**: Applied to stage 3 features (128 channels)
- **DA4**: Applied to stage 4 features (160 channels)

Combines position and channel attention for comprehensive feature refinement.

### 4️⃣ Vessel-Guided Feature Modulation

- Auxiliary vessel segmentation provides structural priors
- Vessel predictions modulate PE features through spatial attention
- Enhances embolism localization within vascular structures

```python
segv_down = F.max_pool3d(segv, kernel_size=2, stride=2).repeat(1, x.shape[1], 1, 1, 1)
x = x * segv_down  # Vessel-guided feature modulation
5️⃣ Multi-Scale ASPP Fusion

Atrous Spatial Pyramid Pooling with multiple dilation rates:

- Dilations: [1, 3, 5, 7, 9]
- Fuses features from four different scales
- Captures emboli of varying sizes

---

## 📂 Project Structure

```bash
DDAG-Net/
├── models/
│   ├── DDAG_Net.py           # Main network architecture
│   ├── StdBlocks.py          # Standard building blocks
│   ├── Attention.py          # Quadruplet attention module
│   ├── block1.py             # DANet attention module
│   └── transformer3d.py      # 3D Transformer implementation
│
├── datasets/
│   ├── dataset.py            # CTPA dataset loader
│   └── transforms.py         # 3D data augmentations
│
├── utils/
│   ├── losses.py             # Segmentation losses
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # 3D visualization tools
│
├── configs/
│   └── ddag_net_config.yaml  # Configuration file
│
├── checkpoints/              # Model checkpoints
├── results/                  # Segmentation results
├── logs/                     # Training logs
│
├── train.py                  # Training script
├── test.py                   # Testing script
├── inference.py              # Inference script
├── requirements.txt          # Dependencies
└── README.md                 # This file
## ⚙️ Requirements

### Environment

- Python >= 3.8
- PyTorch >= 1.9
- CUDA >= 11.0

### Install Dependencies

```bash
### Main Dependencies

```text
torch>=1.9.0
numpy>=1.19.0
scikit-image>=0.18.0
scipy>=1.5.0
SimpleITK>=2.0.0
tqdm>=4.62.0
medpy>=0.4.0


###🚀 Training

##Configuration
Create a configuration file or modify the training script with your settings:

```yaml
# Example config.yaml
model:
  n_channels: 1
  n_classes: 1
  en_blocks: [2, 3, 3, 3]
  de_blocks: [2, 2, 2]
  n_filters: [64, 96, 128, 160]
  stem_filters: 32
  aspp_filters: 32
  norm_type: 'groupnorm'
  act_type: 'ReLU'

training:
  batch_size: 2
  learning_rate: 1e-4
  epochs: 300
  weight_decay: 1e-5

data:
  patch_size: [64, 128, 128]
  stride: [48, 96, 96]
  augmentation: true
Start Training
bash
python train.py \
  --data_root /path/to/ctpa/data \
  --batch_size 2 \
  --lr 1e-4 \
  --epochs 300 \
  --gpu 0
Multi-GPU Training
```bash
python train.py \
  --data_root /path/to/ctpa/data \
  --batch_size 4 \
  --lr 2e-4 \
  --gpu 0,1 \
  --distributed
🧪 Testing
Evaluate model performance on test set:

bash
python test.py \
  --data_root /path/to/test/data \
  --checkpoint checkpoints/best_model.pth \
  --output_dir results/
🔮 Inference
###Run inference on single CTPA volume:

```bash
python inference.py \
  --input /path/to/ctpa/volume.nii.gz \
  --checkpoint checkpoints/best_model.pth \
  --output /path/to/output.nii.gz \
  --save_vessels  # Optional: save vessel segmentation
## 📊 Model Architecture Details

### Encoder Configuration

```text
| Stage   | Input Channels | Output Channels | Blocks | Attention |
|---------|---------------|----------------|--------|-----------|
| Stem    | 1             | 32             | 2      | -         |
| Stage1  | 32            | 64             | 2      | -         |
| Stage2  | 64            | 96             | 3      | -         |
| Stage3  | 96            | 128            | 3      | DA3       |
| Stage4  | 128           | 160            | 3      | DA4       |
### Decoder Configuration

| Stage | Input Channels | Output Channels | Attention |
|-------|---------------|----------------|-----------|
| Dec1  | 160+128       | 128            | Att3      |
| Dec2  | 128+96        | 96             | Att2      |
| Dec3  | 96+64         | 64             | Att1      |
| Final | 64+32+16+8    | 32             | Att4      |

### Transformer Parameters

| Parameter              | Value |
|-----------------------|-------|
| d_model               | 128   |
| nhead                 | 8     |
| Encoder layers        | 6     |
| Decoder layers        | 6     |
| Feedforward dimension | 2048  |
| Dropout               | 0.1   |

## 📈 Loss Functions

The total loss combines multiple components:

```python
L_total = L_dice + L_bce + λ₁ * L_vessel
Dice Loss: Primary segmentation loss

BCE Loss: Pixel-wise binary classification

Vessel Loss: Auxiliary vessel segmentation

Default weights:

λ₁ = 0.5 (vessel segmentation)

📊 Evaluation Metrics
The framework reports the following metrics:

DSC (Dice Similarity Coefficient)

IoU (Intersection over Union)

Sensitivity (True Positive Rate)

Specificity (True Negative Rate)

HD95 (95% Hausdorff Distance)

ASSD (Average Symmetric Surface Distance)

🗃️ Dataset Preparation
Required Format
text
data/
├── train/
│   ├── volume_001.nii.gz
│   ├── volume_001_label.nii.gz
│   └── ...
├── val/
│   ├── volume_001.nii.gz
│   ├── volume_001_label.nii.gz
│   └── ...
└── test/
    ├── volume_001.nii.gz
    ├── volume_001_label.nii.gz
    └── ...
### Preprocessing Steps

```text
1. Resampling: Uniform spacing (1.5×1.5×1.5 mm³)
2. Intensity normalization: Z-score normalization
3. Cropping: Remove background around lungs
4. Patch extraction: 64×128×128 patches with 48×96×96 stride
5. Augmentation: Random flips, rotations, elastic deformations

## 🔥 Key Features

- ✅ **Full 3D processing** - Preserves volumetric information
- ✅ **Transformer backbone** - Global context modeling
- ✅ **Dual attention mechanisms** - Quadruplet + DA attention
- ✅ **Multi-scale ASPP** - Handles variable embolism sizes
- ✅ **Vessel-guided modulation** - Structural prior incorporation
- ✅ **Group normalization** - Batch size independent
- ✅ **Memory efficient** - Optimized for 3D volumes
- ✅ **End-to-end training** - Joint vessel and PE segmentation

📤 Model Outputs
The network forward() returns:

python
{
    "segs": seg,      # Final PE segmentation
    "segvs": segv     # Vessel segmentation branch output
}
Where:

segs denotes the final PE segmentation

segvs denotes the vessel segmentation branch output

###📦 Code Availability
This repository currently serves as the official project page for DDAG-Net.

###🚧 The full source code, pretrained models, and training scripts will be released after the paper is officially accepted.

This ensures compliance with journal and conference submission policies.

###📝 Citation
If you find this work useful for your research, please cite:

bibtex
@article{ddagnet,
  title={DDAG-Net: A novel Dual-Domain Adaptive Decoupling and Arterial Axis-Guided Network for Pulmonary Embolism Detection Segmentation},
  author={},
  journal={},
  year={2026}
}
⭐ Acknowledgements
This project is built upon the PyTorch deep learning framework and open-source medical image segmentation projects.

Special thanks to the medical imaging research community for their valuable contributions.

text

