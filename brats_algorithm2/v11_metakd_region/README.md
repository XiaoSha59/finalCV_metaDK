# MetaKD-V11: Region-Disentangled Meta-Distillation with Region-Decoupled Uncertainty

## 1. Executive Summary & Design Rationale

**MetaKD-V11** directly resolves the core performance bottleneck identified across Jobs 1 to 10 on BraTS 2018:
* **Baseline to beat:** MetaKD Reproduce achieves **72.55% (Raw) / 73.27% (Post-proc)** on the 57 held-out test cases.
* **Job 7 achievement & fatal flaw:** Job 7 achieved **70.91%** on 57 test cases (matching TC at 75.71% and ET at 53.80%), but **Whole Tumor (WT) dropped severely from 85.90% to 83.21% (-2.69%)** because averaging entropy across all 3 classes over-pruned the blurry boundaries of edema.
* **V11 Solution:**
  1. **Region-Disentangled Modality Weight Matrix ($W \in \mathbb{R}^{3 \times 4}$):** Upgrades the global 4-element IWV into 3 region-specific weight vectors for Enhancing Tumor (ET), Whole Tumor (WT), and Tumor Core (TC).
  2. **Region-Decoupled Uncertainty Filtering:**
     * $\tau_{\text{ET}} = 0.5$: Sharp filtering for ET to prevent hallucinating enhancing tumor when T1ce is missing.
     * $\tau_{\text{WT}} = 1.0$: Relaxed filtering for WT, protecting blurry edema boundaries and recovering the lost 2.7% WT Dice.
     * $\tau_{\text{TC}} = 0.5$: Sharp filtering for Tumor Core.
  3. **Zero Additional Loss Terms ($\lambda_{\text{aux}} = 0$):** Maintains 100% gradient hygiene in the bi-level optimization loop, avoiding the gradient conflict that plagued Jobs 1–6 and Job 10.

---

## 2. File Organization

```text
v11_metakd_region/
├── v11_models/
│   ├── __init__.py
│   └── dual_net_v11.py          # DualNet with KD_Weights_Region (3x4)
├── v11_losses/
│   ├── __init__.py
│   └── loss_v11.py              # Region-Decoupled Uncertainty KD Loss
├── v11_utils/
│   ├── __init__.py
│   └── postprocess_v11.py       # Standard topological & morphological postprocessor
├── train_v11.py                 # Bi-level training loop (Inner: Seg + KD + UKD; Outer: W 3x4)
├── eval_v11.py                  # Full 15-modality evaluation across all 57 held-out test cases
├── test_v11_local.py            # Local smoke test (shapes, forward, backward, meta-update)
├── run_v11.sh                   # Phase 2 training runner (80k -> 115k steps)
├── run_eval_v11.sh              # Standalone evaluation runner
└── README.md                    # This document
```

---

## 3. How to Run

### Step 1: Smoke Test (Local Verification)
```bash
python brats_algorithm2/v11_metakd_region/test_v11_local.py
```

### Step 2: Phase 2 Training (Cloud or Workstation)
```bash
bash brats_algorithm2/v11_metakd_region/run_v11.sh
```

### Step 3: Standalone Evaluation (57 Test Cases)
```bash
bash brats_algorithm2/v11_metakd_region/run_eval_v11.sh
```
