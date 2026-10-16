# MetaKD

Reproduction of [Meta-Learned Modality-Weighted Knowledge Distillation](https://arxiv.org/abs/2405.07155) (Wang et al., 2024) on the three public benchmarks in the paper:

| Benchmark | Paper table | This repo |
|---|---|---|
| BraTS 2018 segmentation | Table 1 (15 missing-MRI combinations) | [brats/](brats/README.md) — two-phase 80k → 115k, last checkpoint, 15-combo eval |
| ADNI CN / MCI / AD | Table 2 (C, G, B, I) | [adni/](adni/README.md) — **C / B / I only.** Genomic is omitted |
| AudioVision-MNIST | Table 3 (missing audio / visual rates) | [audiovision/](audiovision/README.md) — held-out 900 / 150 / 450, locked checkpoints |

The authors' code ([billhhh/MetaKD](https://github.com/billhhh/MetaKD)) is BraTS only. Classification follows Algorithm 1, Eq. 2 mean-fill, and Eq. 5 pairwise L1 KD with a learned importance-weight vector. Hyperparameters for ADNI and AudioVision come from the AudioVision supplement: Adam 1e-3, weight decay 1e-2, StepLR every 20 epochs x 0.9; IWV Adam 1e-2. BraTS follows the official `run.sh`: SGD, Nesterov 0.99, lr 1e-2, poly decay, 115k steps, no model selection.

This is **not** a claim that every paper number is recovered. Table 2 G-only 77.32 and C-only 59.65 are not reproducible from the public Flex-MoE tables; Table 1 in the paper is CBICA online (66 cases), while eval here is the local 57-case split.

## Setup

```bash
git clone https://github.com/XiaoSha59/finalCV_metaDK.git
cd finalCV_metaDK
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch with CUDA is required for BraTS. AudioVision and ADNI run on CPU.

## Layout

Each benchmark is one folder with the same three subtrees:

```text
brats/            Table 1
  data/           199 / 29 / 57 split CSVs; place BraTS2018 here
  process/        DualNet, 80k→115k trainer, 15-combo eval
  checkpoints/    phase1_80k / phase2_115k (not in git)

adni/             Table 2, C/B/I
  data/           frozen PTID split + Flex-MoE loaders
  process/        MetaKD trainer
  checkpoints/    locked 3-seed weights

audiovision/      Table 3
  data/           MFCC pairing / MNIST cache
  process/        train + Table 3 runner
  checkpoints/    locked Table 3 weights
```

Raw volumes stay gitignored: `brats/data/BraTS2018/`, `audiovision/data/Audio_MNIST/`, `adni/data/raw/`.

---

## BraTS 2018 (Table 1)

Details: **[brats/README.md](brats/README.md)**.

```bash
# data: brats/data/BraTS2018/MICCAI_BraTS_2018_Data_Training/{HGG,LGG}/

bash brats/process/run_2phase.sh
bash brats/process/run_eval.sh
```

Paper Table 1 averages (ET / TC / WT **63.74 / 79.15 / 86.90**) are from the **online** 66-case set after postprocess. Numbers from `eval_all_modalities.py` are the **local** 57-case test and will not match that row.

---

## ADNI (Table 2, C / B / I)

Full protocol, why G is dropped, and the clinical centering fix: **[adni/README.md](adni/README.md)**.

Loaders are a port of [UNITES-Lab/Flex-MoE](https://github.com/UNITES-Lab/Flex-MoE) (clinical / biospecimen / UCSFFSX7). Genomic is **off by default**. MetaKD and Flex-MoE do not release the Table 2 SNP matrix or `PTID_splits.json`; a Flex-MoE-style PLINK build locally tops out around 43–49% vs paper G **77.32**. Clinical one-hots also have no 59.65-level 3-class signal (logistic ~45, MetaKD ~50). Locked run: drop G, `--center_inputs`, frozen split **2063 / 442 / 442**.

```bash
# data: adni/data/raw/{diagnosis,clinical,biospecimen,image}/  (ADNI license)

python -m adni.process.run_pipeline \
  --allow_generated_split \
  --center_inputs \
  --seeds 0,1,2
```

Accuracy (%) on the frozen split, three seeds, G omitted. Paper column is Table 2 for the same seven rows (mean of those rows: 59.69).

| Combo | Ours (mean ± std) | Paper |
|---|---:|---:|
| C | 50.45 ± 1.26 | 59.65 |
| B | 54.68 ± 2.76 | 56.23 |
| I | 55.13 ± 4.59 | 55.32 |
| C, B | 52.64 ± 2.15 | 63.24 |
| C, I | 58.60 ± 3.53 | 62.75 |
| B, I | 59.65 ± 2.86 | 57.56 |
| C, B, I | 59.65 ± 3.68 | 63.06 |
| Mean | 55.83 ± 2.14 | 59.69 |

Locked weights: `adni/checkpoints/best_adni_model_seed0.pth` (and `seed1`, `seed2`). Diagnosis-onset columns PTCOGBEG, PTADDX, and PTADBEG stay excluded, as in the paper.

---

## AudioVision-MNIST (Table 3)

Data recipe, split, and MFCC steps: **[audiovision/README.md](audiovision/README.md)**. Encoders and MFCCs follow [SMIL](https://github.com/deep-real/SMIL).

```bash
# wavs: audiovision/data/Audio_MNIST/sound/{0-9}/*.wav   MNIST downloads itself

python -m audiovision.process.run_experiments
```

60 epochs, batch 32, seed 42. Locked checkpoints are already in `audiovision/checkpoints/`.

| | 5% | 10% | 15% | 20% | LowerB | UpperB |
|---|---:|---:|---:|---:|---:|---:|
| 3a image test (ours) | 96.22 | 94.67 | 94.44 | 96.22 | 95.33 | 96.67 |
| 3a paper MetaKD | 93.56 | 94.22 | 94.89 | 95.11 | 92.35 | 98.22 |
| 3b audio test (ours) | 85.78 | 88.22 | 87.78 | 87.56 | 90.67 | 96.67 |
| 3b paper MetaKD | 84.67 | 85.56 | 86.89 | 88.44 | 80.44 | 98.22 |

The paper’s audio LowerB (80.44) is SMIL’s 21-clip checkpoint, not this 900-clip training set.

## Citation

```bibtex
@article{wang2024meta,
  title={Meta-Learned Modality-Weighted Knowledge Distillation for Robust Multi-Modal Learning with Missing Data},
  author={Wang, Hu and Hassan, Salma and Liu, Yuyuan and Ma, Congbo and Chen, Yuanhong and Li, Qing and Geng, Jiahui and Wang, Bingjie and Tian, Yu and Xie, Yutong and others},
  journal={arXiv preprint arXiv:2405.07155},
  year={2024}
}
```
