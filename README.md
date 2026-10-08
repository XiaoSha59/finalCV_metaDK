# MetaKD: Meta-Learned Modality-Weighted Knowledge Distillation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2405.07155-b31b1b.svg)](https://arxiv.org/pdf/2405.07155)

This repository contains the reproduction and modular implementation of **MetaKD** (*Meta-Learned Modality-Weighted Knowledge Distillation for Robust Multi-Modal Learning with Missing Data*).

---

## 📖 What Does This Project Do?

In clinical MRI, patients often have **missing modalities** (e.g., missing T1ce due to contrast allergy, or missing Flair/T2 due to scan time constraints). 

### The Core Problem:
Different MRI modalities carry different levels of clinical information:
* **T1ce** is critical for **Enhancing Tumor (ET)** and **Tumor Core (TC)**.
* **Flair** is critical for **Whole Tumor (WT)**.
* When key modalities are missing, conventional deep learning models suffer severe performance drops.

### The MetaKD Solution:
1. **Meta-Learning Importance Weights (IWV):** Dynamically learns a modality importance weight vector $w = [w_{\text{flair}}, w_{\text{t1}}, w_{\text{t1ce}}, w_{\text{t2}}]^\top$ via bi-level optimization.
2. **Modality-Weighted Knowledge Distillation:** Uses the learned weights ($w_i / w_j$) to distill rich feature representations from high-importance modalities to low-importance or missing ones.
3. **Feature Imputation:** Simple and effective mean feature imputation across available modalities.

```
Available Modalities (e.g., Flair + T1) ──> [Shared 3D Encoder] ──> [Feature Imputation & Meta-Distillation] ──> [3D Decoder] ──> (ET, TC, WT Segmentation)
```

---

## 📁 Project Structure

```text
├── data/                        # Raw BraTS 2018 dataset (gitignored)
│   └── BraTS2018/
│       └── MICCAI_BraTS_2018_Data_Training/ (285 labeled cases)
├── datalist/                    # Stratified split lists (Zero data leakage)
│   └── BraTS18/
│       ├── BraTS18_metaTrain.csv   # Meta-training set (~70%, 199 cases)
│       ├── BraTS18_metaVal.csv     # Meta-validation set (~10%, 29 cases)
│       └── BraTS18_test.csv        # Local test set (~20%, 57 cases)
├── datasets/                    # DataLoader & 3D patch data augmentation
│   ├── __init__.py
│   └── brats_dataset.py
├── models/                      # MetaKD model architecture
│   ├── __init__.py
│   ├── dual_net.py              # DualNet + Modality Weighting (IWV)
│   ├── resnet50.py              # 3D ResNet-50 Backbone
│   └── dsn.py                   # Domain-Specific Normalization
├── losses/                      # Loss functions
│   ├── __init__.py
│   └── loss_dual.py             # Pairwise Modality KD Loss + Dice/BCE losses
├── utils/                       # Evaluation metrics & helpers
│   ├── __init__.py
│   ├── metrics.py               # Dice score for ET, TC, WT & table printer
│   └── custom_transforms.py     # 3D spatial/elastic data augmentations
├── train.py                     # Main bi-level meta-learning training script
├── eval.py                      # Single-mode evaluation script
├── eval_all_modalities.py       # Full 15-modality evaluation benchmark
├── split_data.py                # Stratified split generator (70/10/20)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Step-by-Step Reproduction Guide

### 1. Environment Setup

Clone this repository and install dependencies:
```bash
git clone https://github.com/XiaoSha59/finalCV_metaDK.git
cd finalCV_metaDK

python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / GCP Cloud:
source .venv/bin/activate

pip install -r requirements.txt
```

---

### 2. Dataset Preparation & Stratified Split

1. Download the **BraTS 2018** training dataset and place it under `data/BraTS2018/`:
```text
data/BraTS2018/MICCAI_BraTS_2018_Data_Training/
├── HGG/ (210 cases)
└── LGG/ (75 cases)
```

2. Run the stratified split script to create **Train (199 cases)**, **Meta-Val (29 cases)**, and **Test (57 cases)** without data leakage:
```bash
python split_data.py
```

---

### 3. Training the MetaKD Model

MetaKD is trained in **two stages**:

#### Stage 1: Full-Modality Warmup (80,000 iterations)
```bash
python train.py \
  --snapshot_dir=snapshots/BraTS18_warmup/ \
  --input_size=80,160,160 \
  --batch_size=2 \
  --num_steps=80000 \
  --val_pred_every=100 \
  --learning_rate=1e-2 \
  --num_classes=3 \
  --train_list=BraTS18/BraTS18_metaTrain.csv \
  --val_list=BraTS18/BraTS18_metaVal.csv \
  --weight_std=True \
  --reload_from_checkpoint=False
```

#### Stage 2: MetaKD with Random Missing Modality (115,000 iterations)
```bash
python train.py \
  --snapshot_dir=snapshots/BraTS18_MetaKD_115k/ \
  --input_size=80,160,160 \
  --batch_size=2 \
  --num_steps=115000 \
  --val_pred_every=500 \
  --learning_rate=1e-2 \
  --num_classes=3 \
  --train_list=BraTS18/BraTS18_metaTrain.csv \
  --val_list=BraTS18/BraTS18_metaVal.csv \
  --weight_std=True \
  --reload_path=snapshots/BraTS18_warmup/final.pth \
  --reload_from_checkpoint=True \
  --mode=random
```

---

### 4. Evaluating All 15 Missing Modality Combinations

To test the trained model across **all 15 possible combinations of missing modalities** (reproducing Table 1 in the paper) and output the full Dice score table:

```bash
python eval_all_modalities.py --restore_from=snapshots/BraTS18_MetaKD_115k/final.pth
```

#### Expected Evaluation Table:
```text
-----------------------------------------------------------------------------------------------
Modalities             | Enhancing Tumor (ET) | Tumor Core (TC)  | Whole Tumor (WT) | Average   
-----------------------------------------------------------------------------------------------
All (Fl+T1+T1c+T2)     |              79.74%  |          86.20%  |          90.92%  |    85.62%
Fl+T1+T1c              |              77.66%  |          85.36%  |          90.70%  |    84.57%
T1c only               |              76.22%  |          81.98%  |          77.72%  |    78.64%
Flair only             |              47.37%  |          73.01%  |          89.11%  |    69.83%
...
-----------------------------------------------------------------------------------------------
```

---

## ☁️ Cloud & Remote Training Instructions

### 🌟 Training on Google Cloud Platform (GCP - Linux VM)

1. **SSH into your GCP VM with GPU:**
   ```bash
   gcloud compute ssh <YOUR_VM_NAME> --zone=<YOUR_ZONE>
   ```

2. **Verify NVIDIA GPU & CUDA:**
   ```bash
   nvidia-smi
   ```

3. **Clone and Install:**
   ```bash
   git clone https://github.com/XiaoSha59/finalCV_metaDK.git
   cd finalCV_metaDK

   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

4. **Run in Background with `tmux` (Prevents SSH disconnection from stopping training):**
   ```bash
   tmux new -s metakd
   # Run Stage 1 or Stage 2 python train.py command here
   # Detach tmux: Press Ctrl+B, then D
   # Re-attach anytime: tmux attach -t metakd
   ```

5. **Auto-Recovery if Preemptible VM restarts:**
   Simply add `--reload_from_checkpoint=True --reload_path=snapshots/BraTS18_MetaKD_115k/last.pth` to resume from the exact last saved iteration!

---

### 🌟 Training on Kaggle (P100 / T4 GPU)

1. **Create a new Kaggle Notebook:**
   * Go to **Settings** $\rightarrow$ **Accelerator** $\rightarrow$ Select **GPU P100** or **GPU T4 x2**.
   * Go to **Data** $\rightarrow$ **Add Input** $\rightarrow$ Search and add **BraTS 2018 Dataset**.

2. **Setup and Symlink Data:**
   ```python
   # Cell 1: Clone and install
   !git clone https://github.com/XiaoSha59/finalCV_metaDK.git
   %cd finalCV_metaDK
   !pip install -r requirements.txt
   
   # Cell 2: Symlink dataset to data/BraTS2018/
   !mkdir -p data/BraTS2018
   !ln -s /kaggle/input/brats-2018/MICCAI_BraTS_2018_Data_Training data/BraTS2018/MICCAI_BraTS_2018_Data_Training
   !python split_data.py
   ```

3. **Train and Save to `/kaggle/working/`:**
   ```python
   # Cell 3: Stage 1 Warmup
   !python train.py \
     --snapshot_dir=/kaggle/working/snapshots/BraTS18_warmup/ \
     --input_size=80,160,160 \
     --batch_size=2 \
     --num_steps=80000 \
     --val_pred_every=500 \
     --learning_rate=1e-2 \
     --num_classes=3 \
     --train_list=BraTS18/BraTS18_metaTrain.csv \
     --val_list=BraTS18/BraTS18_metaVal.csv \
     --weight_std=True \
     --reload_from_checkpoint=False
   ```

4. **Evaluate directly inside Kaggle Notebook:**
   ```python
   # Cell 4: 15-Modality Evaluation
   !python eval_all_modalities.py --restore_from=/kaggle/working/snapshots/BraTS18_MetaKD_115k/final.pth
   ```

---

## 📑 Citation

```bibtex
@article{wang2024meta,
  title={Meta-Learned Modality-Weighted Knowledge Distillation for Robust Multi-Modal Learning with Missing Data},
  author={Wang, Hu and Hassan, Salma and Liu, Yuyuan and Ma, Congbo and Chen, Yuanhong and Li, Qing and Geng, Jiahui and Wang, Bingjie and Tian, Yu and Xie, Yutong and others},
  journal={arXiv preprint arXiv:2405.07155},
  year={2024}
}
```
