# BraTS 2018 (3D Brain Tumor Segmentation)

3D multi-modal brain tumor segmentation for MetaKD Table 1 ([arXiv:2405.07155](https://arxiv.org/abs/2405.07155)). Architecture, dual-branch feature distillation, and the two-phase 80k → 115k training schedule follow the authors' official repository ([billhhh/MetaKD](https://github.com/billhhh/MetaKD)). 

This document details the dataset, frozen split, preprocessing pipeline, bi-level meta-optimization, post-processing mechanisms (especially regarding Enhancing Tumor), and reproduction instructions across all 15 missing-modality combinations.

---

## 1. Directory Layout

```text
brats/
├── README.md                      # This comprehensive documentation
├── data/
│   ├── dataset.py                 # BraTS 3D patch loader, z-score normalization, random modal drop
│   ├── BraTS18_metaTrain.csv      # 199 cases (70% training split)
│   ├── BraTS18_metaVal.csv        # 29 cases (10% meta-validation split)
│   └── BraTS18_test.csv           # 57 cases (20% held-out test split)
├── process/
│   ├── train.py                   # Bi-level MetaKD training loop (Algorithm 1)
│   ├── eval_all_modalities.py     # 15-modality evaluation with sliding-window & postprocess
│   ├── run_2phase.sh              # Two-phase runner script (80k warmup + 115k MetaKD)
│   ├── run_eval.sh                # 15-combination benchmark runner
│   ├── models/
│   │   ├── dual_net.py            # Dual-Net 3D U-Net backbone
│   │   ├── dsn.py                 # Deeply supervised network branches
│   │   └── resnet50.py            # Feature extractor backbones
│   ├── losses/
│   │   └── loss_dual.py           # Segmentation (Dice + BCE) and pairwise L1 KD loss
│   └── utils/
│       ├── custom_transforms.py   # 3D spatial data augmentations
│       ├── metrics.py             # Dice coefficient & evaluation metrics
│       ├── logger.py              # Training logger & progress trackers
│       └── postprocess.py         # Author's post-processing (connected components + K-Means)
└── checkpoints/                   # Phase 1 (80k) & Phase 2 (115k) snapshots (gitignored)
```

---

## 2. Dataset & Modalities

The benchmark uses the 285 labeled cases (210 HGG / 75 LGG) from the MICCAI BraTS 2018 Training Dataset.

### Modalities (4 MRI sequences per patient)

| Index | Modality | Full Name | Clinical Relevance |
|:---:|:---|:---|:---|
| **0** | **FLAIR** | T2-weighted Fluid Attenuated Inversion Recovery | Hyperintense for whole tumor and peritumoral edema |
| **1** | **T1** | T1-weighted native | Anatomical boundary reference; hypointense necrotic core |
| **2** | **T1ce** | T1-weighted contrast-enhanced (Gadolinium) | **Crucial for active tumor:** hyperintense enhancing tumor rim |
| **3** | **T2** | T2-weighted | Delineates edema and fluid-filled non-enhancing tumor core |

### Target Tumor Subregions (3 overlapping structures)

BraTS annotations contain four raw labels: `0` (background), `1` (necrotic/non-enhancing core), `2` (edema), `4` (enhancing tumor). In clinical evaluation, these are evaluated as 3 nested subregions:

1. **Enhancing Tumor (ET)**: Raw label `4`. Represents active, hyper-vascularized tumor cells.
2. **Tumor Core (TC)**: Raw labels `1 + 4`. Includes necrotic core and active enhancing tumor.
3. **Whole Tumor (WT)**: Raw labels `1 + 2 + 4`. Total tumor mass including infiltrative peritumoral edema.

---

## 3. Data Split & Partitioning

The dataset is partitioned into a fixed **70% / 10% / 20%** split across the 285 cases:

- **Training (`BraTS18_metaTrain.csv`)**: 199 cases (147 HGG / 52 LGG)
- **Meta-Validation (`BraTS18_metaVal.csv`)**: 29 cases (21 HGG / 8 LGG)
- **Test (`BraTS18_test.csv`)**: 57 cases (42 HGG / 15 LGG)

> **Important**: The split files in `brats/data/` are pre-generated and frozen. Never regenerate them, as strict patient isolation is required to prevent data leakage and ensure exact reproducibility.

---

## 4. Preprocessing & Augmentation Pipeline

Processing follows the official MetaKD 3D volumetric pipeline:

1. **Non-Zero Bounding Box Cropping**: Removes uninformative zero-background margins around the skull.
2. **Foreground Z-Score Normalization**: Each modality sequence is normalized independently over non-zero brain voxels:
   $$\hat{x} = \frac{x - \mu_{\text{fg}}}{\sigma_{\text{fg}}}$$
3. **Patch Extraction**: During training, sub-volumes of size **$128 \times 128 \times 128$** are randomly cropped from the volumes.
4. **Data Augmentation**:
   - Random flipping along axial, coronal, and sagittal planes ($p = 0.5$).
   - Random spatial cropping centered on tumor foreground.
5. **Missing-Modality Simulation**: During Phase 2 training (`--mode random`), $0$ to $3$ modalities are randomly masked (filled with zeros) per batch, forcing the model to learn robust shared representations.

---

## 5. MetaKD Architecture & Two-Phase Optimization

### Dual-Net Backbone (`models/dual_net.py`)
MetaKD employs a dual-branch 3D architecture where:
- A primary branch processes available modalities.
- An auxiliary representation is dynamically matched via pairwise knowledge distillation.
- Deep supervision (DSN) outputs multi-scale predictions for intermediate loss computation.

### Bi-Level Optimization (Algorithm 1)
In each training step, optimization alternates between two stages:

1. **Meta-Update of Importance-Weight Vector (IWV)**:
   The importance-weight vector $w^*$ represents the relative information value of each modality. It is updated on the meta-validation set $D_v$ using meta-gradients.
2. **Network Parameter Update**:
   Given fixed $w^*$, the network parameters $\theta$ are trained on the training batch $D_t$ via:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \alpha \cdot \mathcal{L}_{\text{KD}}$$
   where $\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}}$, and the pairwise knowledge distillation loss is:
   $$\mathcal{L}_{\text{KD}} = \sum_{i \neq j} \frac{w_i^*}{w_j^*} \| f_i - f_j \|_1$$
   with $\alpha = 0.01$.

### Two-Phase Training Schedule

| Phase | Steps | Mode | Description |
|---|:---:|:---:|---|
| **Phase 1 (Warmup)** | 0 → 80,000 | `full` | Trains backbone with all 4 modalities present. Establishes strong multi-modal latent representations. |
| **Phase 2 (MetaKD)** | 80,000 → 115,000 | `random` | Reloads Phase 1 checkpoint (`final.pth`). Randomly drops modalities and optimizes both network weights and IWV via bi-level distillation. |

- **Optimizer**: SGD with Nesterov momentum $0.99$, weight decay $10^{-4}$.
- **Learning Rate**: Initial $10^{-2}$, decayed via polynomial schedule: $\text{lr} = \text{lr}_0 \cdot (1 - \frac{\text{step}}{\text{total}})^{0.9}$.
- **Evaluation Rule**: The paper evaluates the **final step (115,000)** checkpoint (`final.pth`), **not** the best validation Dice, following the original protocol.

---

## 6. Post-Processing & The Enhancing Tumor (ET) Dynamics

### Author's Post-Processing (`utils/postprocess.py`)
Raw network predictions are refined prior to official scoring:
1. **Connected Component Filtering**: Removes small isolated ET clusters with volume $< 500$ voxels, eliminating false-positive speckles.
2. **T1ce K-Means Intensity Clustering**: Thresholds the hyperintense T1ce signal inside the tumor core to cleanly distinguish enhancing tumor (ET) from non-enhancing necrosis.

### Why ET Drops Without T1ce
- **Biological Constraint**: Gadolinium contrast enhancement in T1ce is the sole modality that directly visualizes blood-brain barrier breakdown in high-grade gliomas.
- **Observed Behavior**:
  - In combinations **containing T1ce** (e.g., `T1c`, `Fl+T1c`, `All`): ET Dice is high (**75% – 80%**).
  - In combinations **lacking T1ce** (e.g., `Fl`, `T1`, `T2`, `Fl+T2`): raw ET Dice drops to **~45% – 50%**.
- In training logs (`--mode random`), the logged validation ET is an unweighted average across random combinations (both with and without T1ce), naturally hovering around **50% – 58%**. Post-processing and per-combination evaluation in Table 1 reveal the true performance.

---

## 7. Table 1 Benchmark (All 15 Missing-Modality Combinations)

Evaluation follows a sliding-window approach over full volumes with patch size $128 \times 128 \times 128$ and $1/3$ overlap.

### Ground Truth from MetaKD Paper Table 1 vs. Evaluation

| Combination | Modalities Present | Paper ET | Paper TC | Paper WT | Paper Avg |
|:---:|:---|:---:|:---:|:---:|:---:|
| `0` | FLAIR | 47.37 | 73.01 | 89.11 | 69.83 |
| `1` | T1 | 44.94 | 67.16 | 77.47 | 63.19 |
| `2` | T1ce | 76.22 | 81.98 | 77.72 | 78.64 |
| `3` | T2 | 46.93 | 69.72 | 84.44 | 67.03 |
| `0,1` | FLAIR + T1 | 49.64 | 75.66 | 90.23 | 71.84 |
| `0,2` | FLAIR + T1ce | 77.49 | 84.82 | 90.30 | 84.20 |
| `0,3` | FLAIR + T2 | 50.27 | 75.88 | 90.39 | 72.18 |
| `1,2` | T1 + T1ce | 75.41 | 81.85 | 80.59 | 79.28 |
| `1,3` | T1 + T2 | 49.77 | 74.40 | 87.05 | 70.41 |
| `2,3` | T1ce + T2 | 78.24 | 84.84 | 86.18 | 83.09 |
| `0,1,2` | FLAIR + T1 + T1ce | 77.66 | 85.36 | 90.70 | 84.57 |
| `0,1,3` | FLAIR + T1 + T2 | 51.05 | 76.73 | 90.59 | 72.79 |
| `0,2,3` | FLAIR + T1ce + T2 | 75.58 | 85.08 | 90.87 | 83.84 |
| `1,2,3` | T1 + T1ce + T2 | 75.87 | 84.57 | 86.94 | 82.46 |
| `0,1,2,3`| **All Modalities** | **79.74** | **86.20** | **90.92** | **85.62** |
| **Mean** | **Overall 15-Combination Average** | **63.74** | **79.15** | **86.90** | **76.60** |

> **Note on Paper vs. Local Metrics**: The paper's Table 1 summary rows originate from the online CBICA evaluation server (66 unseen cases with unreleased ground truth). Evaluation in `brats/process/eval_all_modalities.py` is conducted on the official held-out local split (57 cases with ground truth), providing reproducible and transparent metrics.

---

## 8. High-Performance Training Optimization (RAM Caching)

Standard disk I/O on cloud persistent disks (e.g., GCP `pd-standard` or `pd-balanced`) causes significant data loading latency (~4.5s per step) due to reading uncompressed 3D NIfTI/Numpy volumes.

To solve this, our pipeline supports **shared-memory RAM caching**:
- All preprocessed sub-volumes are loaded into memory (`/dev/shm` or RAM cache) at initialization.
- **Speedup**: Step time decreases from **4.50s → 1.09s** (~**4.1× acceleration**).
- **Time reduction**: Total Phase 2 execution drops from **~6 days → ~35 hours** on a single NVIDIA L4 / A100 GPU.

---

## 9. Reproduction Instructions

### Prerequisites
```bash
pip install -r requirements.txt
```
PyTorch with CUDA support (VRAM $\ge 16$ GB) is required.

### Step 1: Phase 1 Warmup (80,000 steps)
```bash
python brats/process/train.py \
  --snapshot_dir brats/checkpoints/phase1_80k/ \
  --num_steps 80000 \
  --mode full \
  --batch_size 2 \
  --learning_rate 1e-2 \
  --train_list brats/data/BraTS18_metaTrain.csv \
  --val_list brats/data/BraTS18_metaVal.csv
```

### Step 2: Phase 2 MetaKD (80,000 → 115,000 steps)
```bash
python brats/process/train.py \
  --snapshot_dir brats/checkpoints/phase2_115k/ \
  --reload_from_checkpoint True \
  --reload_path brats/checkpoints/phase1_80k/final.pth \
  --num_steps 115000 \
  --mode random \
  --batch_size 2 \
  --learning_rate 1e-2 \
  --train_list brats/data/BraTS18_metaTrain.csv \
  --val_list brats/data/BraTS18_metaVal.csv
```

### Step 3: 15-Modality Benchmark Evaluation (Table 1)
```bash
bash brats/process/run_eval.sh
```
Or directly via Python:
```bash
python brats/process/eval_all_modalities.py \
  --restore_from brats/checkpoints/phase2_115k/final.pth \
  --test_list brats/data/BraTS18_test.csv \
  --output_csv logs/brats18_table1_results.csv
```
The script evaluates all 15 combinations, generates both raw and post-processed Dice scores, and exports the comparison matrix directly into CSV format.
