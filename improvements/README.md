# Improvements: Plan 1 Extended — Class-Decoupled Uncertainty Knowledge Distillation (CDU-KD)

This directory contains the cleaned, self-contained, and production-grade implementation of **Plan 1 Extended**, our highest-performing proposal for multi-modal brain tumor segmentation under arbitrary missing modalities on the **MICCAI BraTS 2018** benchmark.

---

## 🚀 Key Achievements

* **Benchmarking (CVPR 2023 Leaderboard):**
  * Achieved **72.72% Mean Dice** across all 15 possible combinations of missing modalities.
  * Officially **surpassed RFNet (72.70%)**, one of the strongest published baselines.
  * **Tumor Core (TC): 76.21%** — outperforming both **RFNet (76.00%)** and **MetaKD Reproduce (75.85%)**.
  * **Enhancing Tumor (ET): 56.94%** — outperforming **RFNet (56.60%)**, **SMU-Net (55.00%)**, and **ACN (54.20%)**.
* **Training Efficiency:**
  * Reached full convergence in **51,500 total steps**, cutting down compute by **>55%** compared to MetaKD's 115,000 steps.

---

## 🧠 Motivation & Method

### The Problem in Standard MetaKD
Standard MetaKD estimates predictive uncertainty globally across all segmentation channels:
$$\mathcal{M}(x) = \exp\left(-\frac{\mathcal{H}(p_t(x))}{\tau}\right)$$
This design imposes a severe dilemma:
1. **Whole Tumor (WT / Edema)** has blurry, high-entropy boundaries. Applying global uncertainty filtering penalizes these boundaries and collapses WT accuracy.
2. **Enhancing Tumor (ET)** is extremely small and sparse. If the threshold $\tau$ is set too stringently or uniformly, the student network is starved of ET gradients whenever T1ce is absent.

### The Solution: Class-Decoupled Uncertainty KD (CDU-KD)
CDU-KD completely decouples uncertainty modulation per anatomical region:
$$\mathcal{M}_c(x) = \exp\left(-\frac{\mathcal{H}(p_{t,c}(x))}{\tau_c}\right), \quad c \in \{\text{ET}, \text{WT}, \text{TC}\}$$
- **Channel 0 (ET):** Uses adaptive threshold $\tau_{\text{ET}} = 0.50 \to 0.25$ to open gradient channels for sparse enhancing regions.
- **Channel 1 (WT):** Full preservation with $\tau_{\text{WT}} = \text{None}$ ($\mathcal{M}_{\text{WT}} = 1.0$), safeguarding 100% of the edema boundary.
- **Channel 2 (TC):** Sharp boundary constraint with $\tau_{\text{TC}} = 0.50$.

---

## 📁 Repository Structure

```
improvements/
├── README.md                          # Technical overview and reproduction documentation
├── loss_cdukd.py                      # Core Class-Decoupled Uncertainty KD Loss
├── postprocess.py                     # Official medical domain post-processing pipeline
├── train_plan1_extended.py            # End-to-end training and fine-tuning script
├── eval_plan1_extended.py             # 15-modality evaluation with sliding window inference
├── run_train_pipeline.sh              # Unified automated shell pipeline
└── results/
    ├── results_plan1_extended_72.72.csv   # Official 15-modality evaluation table
    └── cvpr_benchmark_comparison.md      # Head-to-head leaderboard against SOTA methods
```

---

## 📊 Evaluation Results (15 Modality Combinations)

Extracted from [results_plan1_extended_72.72.csv](results/results_plan1_extended_72.72.csv):

| Missing Mode | Modality Description | Enhancing Tumor (ET) | Tumor Core (TC) | Whole Tumor (WT) | Average Dice |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | Flair only | 36.73% | 66.07% | 89.03% | 63.94% |
| `1` | T1 only | 35.23% | 64.72% | 71.00% | 56.98% |
| `2` | T1ce only | 71.04% | 80.31% | 72.65% | 74.67% |
| `3` | T2 only | 37.96% | 63.57% | 81.47% | 61.00% |
| `0,1` | Fl + T1 | 40.39% | 71.70% | 89.67% | 67.25% |
| `0,2` | Fl + T1ce | 73.51% | 85.19% | 89.98% | 82.89% |
| `0,3` | Fl + T2 | 41.55% | 69.13% | 90.18% | 66.95% |
| `1,2` | T1 + T1ce | 72.13% | 81.17% | 74.72% | 76.00% |
| `1,3` | T1 + T2 | 39.59% | 69.69% | 85.07% | 64.78% |
| `2,3` | T1ce + T2 | 72.22% | 82.58% | 85.39% | 80.06% |
| `0,1,2` | Fl + T1 + T1ce | 73.81% | 85.86% | 90.03% | 83.23% |
| `0,1,3` | Fl + T1 + T2 | 42.08% | 72.28% | 90.24% | 68.20% |
| `0,2,3` | Fl + T1ce + T2 | 73.15% | 84.25% | 90.56% | 82.65% |
| `1,2,3` | T1 + T1ce + T2 | 71.54% | 82.08% | 84.48% | 79.37% |
| `0,1,2,3` | All Modalities | 73.13% | 84.53% | 90.56% | 82.74% |
| **Mean** | **Overall 15 Modalities** | **56.94%** | **76.21%** | **85.00%** | **72.72%** |

---

## 🛠️ Reproduction & Usage

### 1. Training from Scratch
To run the full end-to-end 50,000 steps with CDU-KD:
```bash
python improvements/train_plan1_extended.py \
    --snapshot_dir ./snapshots/plan1_extended \
    --data_dir ./data \
    --train_list datalist/BraTS18/BraTS18_metaTrain.csv \
    --val_list datalist/BraTS18/BraTS18_metaVal.csv \
    --num_steps 50000 \
    --phase1_steps 25000 \
    --batch_size 2 \
    --learning_rate 1e-2 \
    --tau_et 0.50 \
    --tau_tc 0.50 \
    --tau_wt -1.0
```

### 2. Fast Fine-Tuning (ET Gradient Unblocking)
To fine-tune 1,500 steps with $\tau_{\text{ET}} = 0.25$:
```bash
python improvements/train_plan1_extended.py \
    --snapshot_dir ./snapshots/plan1_extended_finetuned \
    --reload_from_checkpoint \
    --reload_path ./snapshots/plan1_extended/final.pth \
    --reset_optimizer \
    --start_iters 50000 \
    --num_steps 51500 \
    --learning_rate 1e-4 \
    --tau_et 0.25 \
    --tau_tc 0.50 \
    --tau_wt -1.0
```

### 3. Comprehensive Evaluation
To evaluate across all 15 modality configurations:
```bash
python improvements/eval_plan1_extended.py \
    --data_dir ./data \
    --data_list datalist/BraTS18/BraTS18_test.csv \
    --restore_from checkpoints/plan1_51.5k_finetuned.pth \
    --output_csv improvements/results/results_eval.csv \
    --postprocess true
```
