# RSMIA-Net

**RSMIA-Net: Residual Skip-enhanced Network with Multi-dimensional Interactive Attention for Pulmonary Embolism Detection and Segmentation**

> A 3D pulmonary embolism segmentation and detection framework with attention-guided residual skip enhancement, multi-dimensional interactive attention, context-detail balanced ASPP fusion, and Dual Attention Transformer bottleneck modeling.

---

## 📌 Overview

Pulmonary embolism (PE) is a life-threatening condition that seriously endangers human health. Accurate PE segmentation and detection in **CT pulmonary angiography (CTPA)** remain challenging because emboli are often small, low-contrast, and located inside complex pulmonary vascular structures.

To address the semantic gap between high-level and low-level features and the insufficient spatial context modeling in PE segmentation, we propose **RSMIA-Net**, a **Residual Skip-enhanced Network with Multi-dimensional Interactive Attention**.

RSMIA-Net is built upon a 3D U-Net style encoder-decoder architecture and introduces several complementary modules:

- Attention-Guided Residual Skip (AGRS)
- Residual Skip Enhancement (RSE)
- Multi-dimensional Interactive Quadruplet Attention (MIQA)
- Context-Detail Balanced ASPP (CDB-ASPP)
- Dual Attention (DA) Transformer bottleneck
- Auxiliary vessel-guided feature modulation

The network jointly enhances local detail representation, global context modeling, and vessel-aware PE localization.

---

## 🧠 Network Architecture

The overall architecture of RSMIA-Net is illustrated below:

```text
Input CTPA Volume
   │
CT Channel Selection
   │
Stem Feature Extraction
   ├── 3D ConvBlock
   └── Downsampling ConvBlock
   │
Encoder
   ├── Stage 1: Low-level Feature Extraction
   ├── Stage 2: Hierarchical Feature Extraction
   ├── Stage 3: Deep Semantic Feature Extraction
   └── Stage 4: Bottleneck Feature Extraction
   │
Bottleneck
   ├── Dual Attention Feature Refinement
   ├── 3D Positional Encoding
   └── Transformer Global Context Modeling
   │
Decoder
   ├── Stage 3 Feature Recovery
   │   ├── Upsampling
   │   ├── Dual Attention Skip Refinement
   │   └── Skip Concatenation
   │
   ├── Stage 2 Feature Recovery
   │   ├── Residual Skip Enhancement
   │   ├── Multi-dimensional Interactive Quadruplet Attention
   │   └── Skip Concatenation
   │
   └── Stage 1 Feature Recovery
       ├── Residual Skip Enhancement
       ├── Multi-dimensional Interactive Quadruplet Attention
       └── Skip Concatenation
   │
Context-Detail Balanced ASPP
   ├── Multi-scale Dilated Convolution Branches
   ├── Context Aggregation
   └── Detail-preserving Feature Fusion
   │
Auxiliary Vessel Segmentation Branch
   ├── Vessel Prediction
   ├── Downsampled Vessel Guidance Map
   └── Feature-wise Vessel-guided Modulation
   │
Final PE Segmentation Head
   │
PE Segmentation Output
```

---

## ✨ Key Innovations

### 1️⃣ Attention-Guided Residual Skip (AGRS)

The Attention-Guided Residual Skip module is designed to reduce the semantic gap between encoder and decoder features.

AGRS consists of two parts:

- Residual Skip Enhancement (RSE)
- Multi-dimensional Interactive Quadruplet Attention (MIQA)

This module allows deeper semantic information to enhance shallow feature representations before decoder fusion.

---

### 2️⃣ Residual Skip Enhancement (RSE)

The RSE structure introduces top-down multiplicative guidance and residual fusion into skip connections.

In the implementation, high-level skip modulation is performed as:

```python
s3 = self.block22_up_1(x3)
s3 = x2 * s3
x2 = x2 + s3

s2 = self.block11_up_1(x2)
s2 = x1 * s2
x1 = x1 + s2
```

This design has three effects:

- Transfers deep semantic cues to shallow encoder features
- Preserves original low-level details through residual fusion
- Improves semantic consistency during encoder-decoder feature fusion

---

### 3️⃣ Multi-dimensional Interactive Quadruplet Attention (MIQA)

MIQA captures interdependencies between channel and spatial dimensions from four complementary perspectives.

In the current network, this mechanism is implemented through `QuadrupletAttention` modules:

- `att2` refines intermediate skip features
- `att1` refines shallow skip features
- Spatial-channel interactions are enhanced before skip concatenation
- Fine-grained PE region features are strengthened through spatial interaction

This improves the representation of small emboli and ambiguous boundary regions.

---

### 4️⃣ Context-Detail Balanced ASPP (CDB-ASPP)

CDB-ASPP fuses contextual and detailed information through multi-branch atrous convolution.

The implementation uses `ASPP4` after concatenating multi-scale decoder features:

```python
x = self.aspp(torch.cat([x, x2, x3, x4], dim=1))
```

The module aggregates features from multiple decoder resolutions and uses five dilation settings:

```python
dilations=[1, 3, 5, 7, 9]
```

This improves PE segmentation by balancing:

- Local boundary detail
- Medium-scale embolus morphology
- Large-scale vascular context
- Global-local feature enhancement

---

### 5️⃣ Dual Attention Transformer Bottleneck

RSMIA-Net incorporates a Dual Attention (DA) Transformer module at the bottleneck of the U-Net.

The bottleneck first applies dual attention:

```python
x = self.DA4(x)
```

Then 3D positional encoding is generated and passed into the Transformer:

```python
pos_embed = self.generate_positional_encoding_3d(
    depth, height, width, channels, batch_size
)
hs, memory = self.transformer(src, mask, query_embed, pos_embed)
```

This module enhances global feature modeling and compensates for the limited receptive field of pure convolutional blocks.

---

### 6️⃣ Vessel-guided PE Segmentation

The model includes an auxiliary vessel segmentation branch to provide vascular structure guidance.

```python
segv = self.segheadv(x)
segv_down = F.max_pool3d(segv, kernel_size=2, stride=2).repeat(
    1, x.shape[1], 1, 1, 1
)
x = x * segv_down
seg = self.seghead(x)
```

The auxiliary vessel prediction is downsampled and used to modulate the final feature map. This helps the PE segmentation branch focus on anatomically relevant vascular regions.

---

## 📈 Reported Performance

On the **CAD-PE dataset**, RSMIA-Net achieves strong performance in both PE detection and segmentation.

### PE Detection

- Sensitivity: `0.904`
- FROC score: `0.798`
- AUC: `0.862`

### PE Segmentation

- Dice coefficient: `89.7%`

These results indicate that RSMIA-Net improves PE localization accuracy and detection sensitivity compared with other state-of-the-art methods.

---

## 📂 Project Structure

The following is a recommended GitHub project layout for this model:

```bash
RSMIA-Net/
├── models/
│   ├── RSMIA_Net.py
│   ├── StdBlocks.py
│   ├── Attention.py
│   ├── block1.py
│   └── transformer3d.py
│
├── datasets/
│   ├── dataset.py
│   └── transforms.py
│
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── visualization.py
│
├── checkpoints/
├── results/
├── train.py
├── test.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

### Environment

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Dependencies

```text
torch
numpy
```

Common dependencies for medical image preprocessing and evaluation may include:

```text
scipy
SimpleITK
scikit-image
tqdm
```

---

## 🚀 Training

If the repository is organized with a standard training entry script, training can be started as:

```bash
python train.py
```

Example:

```bash
python train.py \
  --batch_size 2 \
  --lr 1e-4 \
  --epochs 300
```

Elastic deformation-based data augmentation can be used during training to improve robustness and training efficiency on limited datasets.

---

## 🧪 Inference

```bash
python inference.py \
  --ckpt checkpoints/rsmia_net.pth
```

---

## 📊 Evaluation Metrics

The following metrics are used or commonly reported for PE detection and segmentation:

- Dice Similarity Coefficient (DSC)
- Sensitivity
- FROC score
- AUC
- Intersection over Union (IoU)
- 95% Hausdorff Distance (HD95)

---

## 🗃️ Dataset

RSMIA-Net is designed for **CT pulmonary angiography (CTPA)** datasets, including CAD-PE style PE segmentation and detection data.

Typical preprocessing includes:

- Resampling
- Intensity normalization
- Lung or vessel region cropping
- 3D patch extraction
- Elastic deformation augmentation

---

## 🔥 Features

- 3D end-to-end PE segmentation
- Residual skip-enhanced encoder-decoder fusion
- Multi-dimensional interactive attention
- Dual Attention Transformer bottleneck
- Context-detail balanced ASPP fusion
- Auxiliary vessel-guided feature modulation
- Strong PE detection and segmentation performance on CAD-PE

---

## 📤 Model Outputs

The current network `forward()` returns:

```python
{
    "segs": seg,
    "segvs": segv,
}
```

Where:

- `segs` denotes the final PE segmentation output
- `segvs` denotes the auxiliary vessel segmentation output

---

## 🧩 Core Implementation

The core model is implemented as:

```python
class Net(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        en_blocks=[2, 3, 3, 3],
        de_blocks=[2, 2, 2],
        n_filters=[64, 96, 128, 128],
        stem_filters=32,
        aspp_filters=32,
        norm_type="groupnorm",
        head_norm="groupnorm",
        act_type="ReLU",
    ):
        ...
```

Default configuration:

- Input channels: `1`
- Output classes: `1`
- Encoder blocks: `[2, 3, 3, 3]`
- Decoder blocks: `[2, 2, 2]`
- Encoder filters: `[64, 96, 128, 128]`
- Stem filters: `32`
- ASPP filters: `32`
- Normalization: `groupnorm`

---

## 📦 Code Availability

This repository currently serves as the project page for RSMIA-Net.

> 🚧 Full training scripts, pretrained weights, dataset preprocessing code, and inference utilities can be released according to the final project plan.

---

## ⭐ Acknowledgements

This project is built upon the PyTorch deep learning framework and open-source medical image segmentation components.

Special thanks to the medical imaging research community for their valuable contributions.
