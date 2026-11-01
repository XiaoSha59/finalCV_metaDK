# AudioVision-MNIST Uncertainty-Aware Distillation (U-MetaKD)

This module implements **Uncertainty-Modulated Cross-Modal Distillation** for image-audio digit recognition on the AudioVision-MNIST benchmark (MetaKD Table 3).

---

## 1. Motivation: Why Uncertainty on AudioVision?
* **Problem in Standard MetaKD:**
  * Baseline MetaKD computes an unweighted L1 distance between visual and audio features:
    $$L_{\text{KD}} = \left(\frac{w_v}{w_a} + \frac{w_a}{w_v}\right) \| f_v - f_a \|_1$$
  * Some audio clips suffer from background noise or ambient artifacts, and some MNIST digit strokes are ambiguous (e.g., a messy digit '7' that resembles '1').
  * Forcing the student to rigidly match the feature representations of ambiguous pairs pollutes cross-modal feature alignment.
* **Solution in U-MetaKD:**
  * Compute the full-modality teacher's predictive Shannon entropy:
    $$\mathcal{H}(p_T) = - \sum_{c=0}^9 p_{T,c} \log(p_{T,c} + \epsilon)$$
  * Modulate the cross-modal distillation by the confidence weight $u_k = \exp(-\mathcal{H}(p_{T,k}) / \tau)$.
  * High-confidence, clearly spoken/written digits receive $u_k \approx 1.0$, while noisy/garbled inputs are automatically down-weighted ($u_k \to 0$).

---

## 2. File Organization
```
improvements/audiovision_uncertainty/
├── loss_audiovision_ukd.py     # Uncertainty cross-modal distillation loss
├── train_audiovision_ukd.py    # Training runner for Table 3a & 3b (5%, 10%, 15%, 20%)
├── audiovision_ukd_table3.csv  # Official Table 3 test results
└── README.md                   # Technical documentation
```

---

## 3. How to Run
```bash
# Run single setting (e.g. missing audio, 10% available rate)
python -c "from improvements.audiovision_uncertainty.train_audiovision_ukd import train_metakd_audiovision_ukd; train_metakd_audiovision_ukd('audio', 0.10, epochs=60)"

# Run full Table 3 benchmark
python -m improvements.audiovision_uncertainty.train_audiovision_ukd
```

---

## 4. Key Results Summary (Table 3)

| Setting | 5% | 10% | 15% | 20% |
| :--- | :---: | :---: | :---: | :---: |
| **Table 3a (Image Test)** | 93.78% | 95.78% | 94.89% | 95.33% |
| **Table 3b (Audio Test)** | 86.89% | 86.22% | **89.56%** | **88.67%** |

* Note: Table 3b beats original Paper MetaKD at 15% rate (89.56% vs 88.44%).

