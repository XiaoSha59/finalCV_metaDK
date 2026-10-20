# Algorithm 2: Region-Aware Shared-Specific Meta-Knowledge Distillation (MetaKD-v2)

## 1. Executive Summary & Objective

This repository contains the advanced algorithmic formulation and experimental roadmap upgrading **MetaKD** (Wang et al., 2024) for incomplete multi-modal brain tumor segmentation on BraTS 2018.

### Core Bottlenecks of Original MetaKD (Algorithm 1):
1. **Global Weight Dilemma:** A single modality importance weight vector $w \in \mathbb{R}^4$ is learned globally across the entire brain volume and across all heterogeneous tumor sub-regions ($\text{ET}, \text{TC}, \text{WT}$), ignoring anatomical modality specificity (e.g., T1ce is clinically decisive for Enhancing Tumor, while FLAIR/T2 characterize Edema/Whole Tumor).
2. **Entangled Knowledge Contamination:** Student forced to mimic 100% of Teacher's feature representations, corrupting student representations with "hallucinated" modality-specific details that missing modalities cannot provide.
3. **Sub-optimal Missing Imputation:** Simple channel-wise mean imputation introduces background blur and noise into feature aggregations.
4. **Dominance of Healthy Brain Background:** Global KD loss computes $L_1$ or $L_2$ over all voxels, where >95% are non-tumor background, diluting gradients intended for tumor pathology.

---

## 2. Theoretical Foundations & Reference Papers

| Component | Source Paper | Key Concept Adopted | Role in Proposed Algorithm |
| :--- | :--- | :--- | :--- |
| **Framework** | **MetaKD** (Wang et al., arXiv:2405.07155, 2024) | Bi-level optimization (Inner Loop on Train, Outer Loop on Val) with Teacher-Student KD | Foundation optimization framework (Alg. 1, Eq. 2, Eq. 5, Eq. 6). |
| **Region-Awareness** | **RFNet** (Ding et al., ICCV 2021) | Region-Aware Fusion for heterogeneous sub-tumors ($\text{ET}, \text{TC}, \text{WT}$) | Disentangling 1 global weight into 3 region-specific weight vectors $w_{\text{ET}}, w_{\text{TC}}, w_{\text{WT}}$. |
| **Disentanglement** | **ShaSpec** (Wang et al., CVPR 2023) | Shared-Specific Feature Decomposition ($r_m, s_m$) via Orthogonal and Shared Alignment losses | Constraining KD exclusively to shared representations $r$, isolating specific features $s$. |
| **Zero-Imputation** | **mmFormer** (Zhang et al., MICCAI 2022) | Incomplete modality handling via selective zero-imputation | Gating missing modality specific tokens/features to zero: $s_m = 0$ if modality $m$ is missing. |

---

## 3. The 3 Experimental Tracks

To evaluate each contribution rigorously and enable comprehensive ablation studies, the evaluation is structured into **3 distinct experimental tracks**:

```
                              ┌────────────────────────────────────────┐
                              │          MetaKD 2-Phase Setup          │
                              └────────────────────────────────────────┘
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 ▼                                ▼                                ▼
    ┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
    │      EXPERIMENT 1       │      │      EXPERIMENT 2       │      │      EXPERIMENT 3       │
    │  (Fast Asymmetric KD)   │      │    (Without ShaSpec)    │      │ (Symmetric from Scratch)│
    ├─────────────────────────┤      ├─────────────────────────┤      ├─────────────────────────┤
    │ • Phase 1: REUSE        │      │ • Phase 1: REUSE        │      │ • Phase 1: RETRAIN      │
    │   existing checkpoint   │      │   existing checkpoint   │      │   0 -> 80k steps        │
    │ • Phase 2:              │      │ • Phase 2:              │      │ • Phase 2:              │
    │   + RFNet (3 w vectors) │      │   + RFNet (3 w vectors) │      │   + Full ShaSpec        │
    │   + mmFormer (0-impute) │      │   + mmFormer (0-impute) │      │     (Teacher & Student) │
    │   + Masked KD           │      │   + Masked KD           │      │   + RFNet + mmFormer    │
    │   + ShaSpec on Student  │      │   - NO ShaSpec (no r/s) │      │     Symmetric KD        │
    └─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

### Track 1: Phase 2 with Asymmetric ShaSpec (Recommended First Step)
* **Goal:** Full Algorithm 2 capabilities executed efficiently without retraining Phase 1.
* **Teacher Network:** Reuses pre-trained full-modality checkpoint (`phase1_80k/final.pth`).
* **Student Network:** Augmented with a lightweight Shared/Specific projection head.
  * Shared feature $r_m$ aligns with Teacher via a $1\times 1\times 1$ projection layer.
  * Orthogonal loss ($\mathcal{L}_{\text{ortho}}$) and Shared Consistency loss ($\mathcal{L}_{\text{shared}}$) active.
  * 3 region weights: $w_{\text{ET}}, w_{\text{TC}}, w_{\text{WT}} \in \mathbb{R}^4$.
  * Region-masked KD loss active.
  * Zero-imputation on missing specific channels ($s_m = 0$).

### Track 2: Phase 2 without ShaSpec (Clean Ablation / Direct Upgrade)
* **Goal:** Direct ablation to quantify the isolated gain of Region-Aware Weights + Masked KD + Zero-imputation over vanilla MetaKD.
* **Teacher Network:** Reuses pre-trained checkpoint (`phase1_80k/final.pth`).
* **Student Network:** Uses standard backbone (no shared/specific channel splitting).
  * 3 region weights: $w_{\text{ET}}, w_{\text{TC}}, w_{\text{WT}}$.
  * Region-masked KD loss applied directly to feature maps.
  * Modality features zero-imputed when absent.

### Track 3: Full Symmetric ShaSpec Trained from Scratch
* **Goal:** Complete theoretical symmetry where both Teacher and Student possess explicit Shared/Specific architectures.
* **Phase 1 Retraining (0 → 80,000 steps):**
  * Train full-modality model with ShaSpec heads, optimizing $\mathcal{L}_{\text{seg}} + \lambda_1 \mathcal{L}_{\text{ortho}} + \lambda_2 \mathcal{L}_{\text{shared}}$.
* **Phase 2 Training (80,000 → 115,000 steps):**
  * Student extracts $r^S_m, s^S_m$. Teacher provides $r^T_m, s^T_m$.
  * Pure shared-to-shared distillation: $r^S \leftarrow r^T$.
  * Region weights $w_r$ modulate regional shared-feature fusion.

---

## 4. Formal Mathematical Formulation: Algorithm 2

### Mathematical Notations:
* Modalities: $\mathcal{M} = \{0: \text{FLAIR}, 1: \text{T1}, 2: \text{T1ce}, 3: \text{T2}\}$.
* Available modalities in current sample: $\mathcal{A} \subseteq \mathcal{M}$.
* Sub-regions: $r \in \{\text{ET}, \text{TC}, \text{WT}\}$.
* Modality importance weights: $\mathbf{W} = [\mathbf{w}_{\text{ET}}, \mathbf{w}_{\text{TC}}, \mathbf{w}_{\text{WT}}] \in \mathbb{R}^{3 \times 4}$.
* Binary spatial masks for sub-regions: $M_r \in \{0, 1\}^{D \times H \times W}$.

---

### Algorithm 2: Region-Aware Shared-Specific Meta-Knowledge Distillation

```text
Input:
  - Training dataset D_train, Validation dataset D_val
  - Frozen Teacher model T with parameters Theta_T (from Phase 1)
  - Student model S with encoder Theta_enc, decoder Theta_dec, projection heads
  - Learning rates: alpha (Student optimizer), beta (Meta weight optimizer kd_optim)
  - Loss balancing hyper-parameters: lambda_KD, lambda_ortho, lambda_shared

Initialize:
  - Student parameters Theta_S = {Theta_enc, Theta_dec} loaded from Phase 1 checkpoint
  - Region-specific weight vectors: w_ET, w_TC, w_WT in R^4, initialized uniformly or randomly
  - Optimizer for Theta_S (SGD with momentum 0.99, lr = alpha)
  - Optimizer for W = {w_ET, w_TC, w_WT} (Adam, lr = beta)

FOR step = 80,000 to 115,000 DO:

    ======================================================================
    1. INNER LOOP: Student Optimization (on Mini-Batch B_train)
    ======================================================================
    Sample batch (X, Y) from D_train with random modality subset A subseteq {0, 1, 2, 3}
    Extract ground-truth binary masks for sub-regions:
        M_ET = (Y == 1 | Y == 3),  M_TC = (Y == 1 | Y == 2 | Y == 3),  M_WT = (Y > 0)

    For each available modality m in A:
        F_m = Encoder(X_m)
        (r_m, s_m) = ShaSpec_Heads(F_m)    # [ShaSpec: Shared-Specific Split]

    # mmFormer: Zero-Imputation for missing specific features
    For each missing modality m not in A:
        s_m = 0                            # [mmFormer: Zero-Imputation]

    # Shared Feature Aggregation:
    r_agg = Mean_{m in A} (r_m)

    # Feature Reconstruction for Decoder:
    F_fused = Concat(r_agg, s_0, s_1, s_2, s_3)
    Y_pred = Decoder(F_fused)

    # Supervised Segmentation Loss:
    L_seg = Dice(Y_pred, Y) + BCE(Y_pred, Y)

    # ShaSpec Feature Constraints:
    L_shared = Sum_{i, j in A, i != j} || r_i - r_j ||_2^2
    L_ortho  = Sum_{i in A} | <r_i, s_i> | / (||r_i||_2 * ||s_i||_2 + eps)

    # Teacher Forward Pass (Full Modality, Frozen):
    With torch.no_grad():
        F_T = Teacher(X_full)

    # Region-Aware Masked Knowledge Distillation Loss:
    L_KD = 0
    For region r in {ET, TC, WT}:
        w_norm_r = Softmax(w_r)
        For modality m in A:
            # Distill only inside region mask M_r [RFNet + Masking]:
            diff = (r_m - F_T_projected)^2
            L_KD += w_norm_r[m] * ( (diff * M_r).sum() / (M_r.sum() + eps) )

    # Total Student Loss:
    L_total = L_seg + lambda_KD * L_KD + lambda_ortho * L_ortho + lambda_shared * L_shared

    # Update Student Parameters:
    Theta_S <- Theta_S - alpha * Grad_{Theta_S}(L_total)


    ======================================================================
    2. OUTER LOOP: Meta-Update of Region Weights (Every val_every steps)
    ======================================================================
    IF step % val_pred_every == 0 THEN:
        Sample validation volume (X_val, Y_val) from D_val

        # Execute single sliding-window inference with current Student:
        Y_val_pred = Predict_Sliding(Student, X_val, val_weights = W)

        # Compute isolated sub-region validation losses:
        L_val_ET = Dice(Y_val_pred[:, ET], Y_val[:, ET]) + BCE(Y_val_pred[:, ET], Y_val[:, ET])
        L_val_TC = Dice(Y_val_pred[:, TC], Y_val[:, TC]) + BCE(Y_val_pred[:, TC], Y_val[:, TC])
        L_val_WT = Dice(Y_val_pred[:, WT], Y_val[:, WT]) + BCE(Y_val_pred[:, WT], Y_val[:, WT])

        # Backward propagation into region-specific weight vectors:
        kd_optim.zero_grad()
        L_meta_val = L_val_ET + L_val_TC + L_val_WT
        L_meta_val.backward()

        # Update region importance weights:
        w_ET <- w_ET - beta * Grad_{w_ET}(L_val_ET)
        w_TC <- w_TC - beta * Grad_{w_TC}(L_val_TC)
        w_WT <- w_WT - beta * Grad_{w_WT}(L_val_WT)
        kd_optim.step()

    END IF

END FOR

Output: Optimized Student model Theta_S and region weight matrix W.
```

---

## 5. Implementation Roadmap & Milestones

1. **Phase 2 Checkpoint Verification:** Verify `d:\MetaKD\brats\checkpoints\phase1_80k\final.pth` validity.
2. **Build Experiment 2 (Ablation without ShaSpec):**
   * Edit `dual_net.py`: Upgrade `KD_Weights` to `(3, 4)` and install masked KD + mmFormer zero-imputation.
   * Edit `train.py`: Update `validate()` to optimize 3 region loss objectives into `kd_weights`.
   * Run 1 epoch dry-run test to verify GPU VRAM and gradient computation.
3. **Build Experiment 1 (Phase 2 with ShaSpec):**
   * Implement Shared/Specific projection heads and orthogonal/shared alignment loss functions.
4. **Benchmark & Dual Evaluation (Raw & Adaptive Post-Processing):**
   * Standalone evaluation runner: `brats_algorithm2/eval_alg2.py`
   * Smart morphological & topological post-processing: `brats_algorithm2/alg2_utils/postprocess_alg2.py`
     * Enforces clinical nesting hierarchy: $\text{ET} \subseteq \text{TC} \subseteq \text{WT}$.
     * Removes only minor scanner dust (< 30 voxels), never wiping out real small enhancing tumors.
     * Functions independently of T1ce availability (no dependency on missing modality).
   * Generates two side-by-side CSV tables:
     * `alg2_table1_raw.csv`: Pure architectural predictions directly from the network.
     * `alg2_table1_postproc.csv`: Predictions refined with adaptive morphological cleaning.


---

## 6. Strict Engineering Guarantees & Stability Safeguards (Zero-OOM & Zero-NaN Protocol)

To eliminate any risk of resource leaks, negative losses, numerical instabilities (`NaN`), or GPU compute waste, the implementation strictly enforces the following engineering protocols:

### A. Zero-NaN & Numerical Stability Guarantees
1. **Safe Mask Division:** In 3D random patch cropping, sub-region masks ($M_{\text{ET}}, M_{\text{TC}}, M_{\text{WT}}$) can be empty ($M_r.\text{sum}() = 0$). Computing $\frac{\sum \text{diff} \cdot M_r}{\sum M_r}$ directly causes $0/0 = \text{NaN}$.
   * **Mandatory Guard:**
     ```python
     mask_sum = M_r.sum()
     if mask_sum > 0:
         loss_kd_r = (diff * M_r).sum() / (mask_sum + 1e-7)
     else:
         loss_kd_r = torch.tensor(0.0, device=diff.device)
     ```
2. **Orthogonal Loss Boundedness (Non-negative & Finite):**
   * Cosine similarity calculations must be protected with epsilon in denominators: $\frac{|\langle r, s \rangle|}{\|r\|_2 \cdot \|s\|_2 + 1e-7}$.
   * Values must be explicitly clamped within $[-1.0, 1.0]$ via `torch.clamp` before loss reduction to prevent numerical precision drift beyond the unit sphere. Loss is strictly non-negative ($\ge 0$).

### B. GPU VRAM & Memory Leak Prevention
1. **Sliding-Window Computational Graph Sanitation:**
   * In `predict_sliding()`, intermediate feature representations across all 18–24 tiles must NOT accumulate detached tensors in VRAM.
   * `torch.cuda.empty_cache()` and explicit `del` calls for predictions and temporary volumes must be strictly preserved after every validation step.
2. **Environment Variable Integrity:**
   * Preserve `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` in all shell scripts to prevent CUDA memory fragmentation.

### C. Pipeline Non-Invasiveness (Zero Regression Guarantee)
1. **Zero Modifications to Existing Optimizations:**
   * Preserve all existing DataLoader workers (`NUM_WORKERS=6`), RAM cache mechanics, random mirror, scaling, and bounding box localization logic.
   * Preserve checkpoint resume semantics (`last.pth` resume, `final.pth` validation, optimizer state restoration).
2. **Safe Backup Protocol:**
   * Before modifying any active file (`dual_net.py`, `train.py`), an exact byte-for-byte backup copy (`.backup_baseline`) will be created in place.

### D. Continuous Execution & Automated Cloud Shutdown Protocol (GPU Cost Protection)
To ensure the training runs uninterruptedly without human supervision while guaranteeing zero wasted cloud GPU billing:
1. **Continuous Execution & Fault-Tolerant Auto-Resume:**
   * Training scripts run under background process supervision (`nohup` or `systemd`).
   * If any temporary OS/driver disruption occurs, the script automatically detects the latest checkpoint (`checkpoints/.../last.pth`) and resumes with `--reload_from_checkpoint True --reload_path "$RELOAD_PATH"` without restarting from step 0.
2. **Automated Power-Off Trigger (On Completion or Fatal Error):**
   * Uses bash `trap` hooks and exit handlers:
     ```bash
     # Auto-shutdown handler upon exit (whether SUCCESS or ERROR)
     cleanup_and_poweroff() {
         EXIT_CODE=$?
         echo "[$(date)] Training script terminated with exit code: $EXIT_CODE"
         sync
         sleep 30
         if [ "$ENABLE_SHUTDOWN" = "true" ]; then
             echo ">>> Auto-shutdown triggered to protect GPU budget. Powering off..."
             sudo /usr/sbin/poweroff -f || sudo shutdown -h now
         fi
     }
     trap cleanup_and_poweroff EXIT
     ```
   * **Guarantee:** If training finishes successfully, the VM powers off immediately after syncing weights to disk. If training crashes with an unrecoverable exception, logs are flushed to disk, and the VM still powers off automatically to avoid idling and wasting cloud budget.


