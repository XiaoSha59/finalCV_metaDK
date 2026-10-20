# DualNet-V3: Non-Invasive ShaSpec Disentanglement & Soft Region-Aware KD

## 1. Executive Summary & Design Rationale

DualNet-V3 resolves the two critical failure modes identified during previous experiments:

1. **Failure Mode 1 (Run 1 - ShaSpec Bottleneck Collapse to 50.60%):**
   - *Problem:* Randomly initialized projection layers inserted directly between Phase 1 pre-trained encoder and decoder broke the continuous feature manifold.
   - *Solution:* **Non-Invasive Auxiliary ShaSpec Head.** The main forward flow directly feeds pre-trained features into `shared_dec` with author's proven Mean-Imputation. ShaSpec heads operate exclusively in an auxiliary parallel branch to guide distillation loss, causing **zero feature distribution shift**.

2. **Failure Mode 2 (Run 2 - Hard Masked KD Causing Background Blindness to 65.50%):**
   - *Problem:* Distilling exclusively inside the binary tumor mask ($r\_mask$) zeroed out gradients on 97% healthy brain background. Without Teacher supervision on normal brain parenchyma, the Student generated massive false positives on contrast-poor modalities (e.g. T1 only dropped to 34%).
   - *Solution:* **Soft Region-Aware KD Loss.** Healthy brain background receives weight 1.0 (actively teaching Student to avoid false positives), while tumor sub-regions receive soft boosted gradient emphasis ($1.0 + 2.0 \times M_{\text{tumor}}$).

3. **Loss Scale Normalization:**
   - $\lambda_{\text{kd}} = 0.1$ strictly aligns with author's `--alpha 0.1`.
   - Spatial loss is normalized by total spatial weight $\times$ feature channels, guaranteeing numerically stable, bounded gradients.

---

## 2. Directory Layout (100% Isolated in `v3_non_invasive/`)

```text
v3_non_invasive/
├── models/
│   ├── __init__.py
│   └── dual_net_v3.py         # DualNet_V3 architecture with Non-Invasive ShaSpec & Soft Region KD
├── train_v3.py                # Bi-level training loop (Inner: Seg + KD + ShaSpec; Outer: W_3x4)
├── eval_v3.py                 # 15-modality sliding-window benchmark with adaptive postprocessing
├── test_v3_forward.py         # Unit test verifying shapes, gradients, and zero-NaN safety
├── run_v3.sh                  # Execution runner for Phase 2 finetuning (80k -> 115k steps)
└── run_eval_v3.sh             # Benchmark evaluation runner for all 15 incomplete modality subsets
```

---

## 3. Execution Commands

### Training Phase 2 (80,000 -> 115,000 steps):
```bash
bash brats_algorithm2/v3_non_invasive/run_v3.sh
```

### Evaluation across 15 Modality Combinations (Table 1):
```bash
bash brats_algorithm2/v3_non_invasive/run_eval_v3.sh
```
