# ADNI Uncertainty-Aware Distillation (U-MetaKD / CDU-KD)

This module extends the **MetaKD framework on ADNI** with **Class-Decoupled Uncertainty Knowledge Distillation (CDU-KD)**.

---

## 1. Motivation: Why Uncertainty on ADNI?
* **Problem in Standard MetaKD:**
  * Baseline MetaKD computes pairwise L1 loss between modality latent vectors for every patient equally:
    $$L_{\text{KD}} = \sum_{i,j} \frac{w_i^*}{w_j^*} \| f_i - f_j \|_1$$
  * When a patient is in the borderline/transitional state (**MCI: Mild Cognitive Impairment**), the representation naturally carries high ambiguity. Forcing the student to rigidly match conflicting features introduces gradient noise.
* **Solution in CDU-KD:**
  * Measure the **Shannon Entropy** $\mathcal{H}(p_k)$ of the teacher's prediction:
    $$\mathcal{H}(p_k) = - \sum_{c=1}^3 p_{k,c} \log(p_{k,c} + \epsilon)$$
  * Modulate the distillation loss by confidence weight $u_k = \exp(-\mathcal{H}(p_k) / \tau_c)$.
  * **Class-Decoupling:**
    * $\tau_{\text{CN}} = 0.50$ (clean cognitively normal profile).
    * $\tau_{\text{MCI}} = 0.80$ with bounded minimum confidence ($u_k \ge 0.50$) to avoid starving the transitional MCI class.
    * $\tau_{\text{AD}} = 0.50$ (clear Alzheimer's pathology).

---

## 2. File Organization
```
improvements/adni_uncertainty/
├── loss_adni_ukd.py            # Core Class-Decoupled Uncertainty loss module
├── train_adni_ukd.py           # End-to-end training pipeline (supports 3-mod and 4-mod)
├── adni_table2_ukd_3mod.csv    # Official Table 2 results for clean 3-modality (C, B, I)
├── adni_table2_ukd_4mod.csv    # Official Table 2 results for full 4-modality (C, G, B, I)
└── README.md                   # Technical documentation
```

---

## 3. How to Run

### A. 3-Modality Clean Setup (C, B, I - 7 Scenarios)
```bash
python -m improvements.adni_uncertainty.train_adni_ukd --allow_generated_split --center_inputs --drop_g --epochs 60 --seeds 0,1,2
```

### B. Full 4-Modality Setup (C, G, B, I - 15 Scenarios)
```bash
python -m improvements.adni_uncertainty.train_adni_ukd --allow_generated_split --center_inputs --no-drop_g --epochs 60 --seeds 0,1,2
```

---

## 4. Key Results Summary

### 3-Modality Clean (C, B, I)
* **MRI (I)**: Test Acc **57.32%** (+2.00% vs Paper), Macro-F1 **56.58%** (+17.53% vs Paper).
* **Bio + MRI (B, I)**: Test Acc **60.18%** (+2.62% vs Paper), Macro-F1 **60.33%** (+20.79% vs Paper).
* **C, B, I**: Test Acc **60.48%**, Macro-F1 **59.93%** (+17.28% vs Paper).

### Full 4-Modality (C, G, B, I)
* **Genomics (G)**: 41.86% (accurately reflects 40,956 raw SNPs without data leakage).
* **Full Multi-modal (C, G, B, I)**: Test Acc **60.41%**, Macro-F1 **59.35%** (+2.12% vs Paper).
* U-MetaKD successfully shields the multi-modal network from genomic noise.

