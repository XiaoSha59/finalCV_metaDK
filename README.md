# MetaKD

Reproduction of [Meta-Learned Modality-Weighted Knowledge Distillation](https://arxiv.org/abs/2405.07155) (Wang et al., 2024) on the three public benchmarks in the paper:

| Benchmark | Paper table | This repo |
|---|---|---|
| BraTS 2018 segmentation | Table 1 (15 missing-MRI combinations) | Two-phase 80k → 115k trainer, last checkpoint, 15-combo eval |
| ADNI CN / MCI / AD | Table 2 (C, G, B, I) | **C / B / I only.** Genomic is omitted; see [adni_flexmoe/README.md](adni_flexmoe/README.md) |
| AudioVision-MNIST | Table 3 (missing audio / visual rates) | Held-out 900 / 150 / 450 split, locked checkpoints |

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

```text
train.py  eval.py  eval_all_modalities.py   # BraTS
run_brats_2phase.sh  run_eval_pipeline.sh   # 80k warmup → 115k MetaKD → Table 1
datasets/  models/  losses/  utils/
datalist/BraTS18/                           # 199 / 29 / 57 split (do not regenerate)

run_adni_pipeline.py
adni_flexmoe/                               # C/B/I loaders, frozen split, 3-seed weights
  README.md  PTID_splits.json  checkpoints/

audiovision/                                # Table 3
  README.md  dataset.py  models.py  train.py  run_experiments.py  checkpoints/
```

Raw data is gitignored: `data/BraTS2018/`, `data/Audio_MNIST/`, `adni_raw_flexmoe/`.

---

## BraTS 2018 (Table 1)

285 labeled cases (210 HGG / 75 LGG). Place the training tree at:

```text
data/BraTS2018/MICCAI_BraTS_2018_Data_Training/{HGG,LGG}/
```

The lists in `datalist/BraTS18/` are already the 70 / 10 / 20 split (199 / 29 / 57). Do not run `split_data.py` unless you intend to change it.

Training is the official two-phase schedule. Phase 1: all four modalities, 80k steps. Phase 2: reload `final.pth`, `--mode=random` (drop 0–3 modalities), continue to 115k. The paper evaluates the **last** checkpoint, not the best validation Dice.

```bash
bash run_brats_2phase.sh          # writes snapshots/BraTS18_phase{1_80k,2_115k}/
bash run_eval_pipeline.sh         # 15 combinations, raw + author postprocess CSVs
```

Equivalent Python (phase 2 only, after warmup):

```bash
python train.py \
  --snapshot_dir snapshots/BraTS18_phase2_115k/ \
  --reload_from_checkpoint True \
  --reload_path snapshots/BraTS18_phase1_80k/final.pth \
  --num_steps 115000 \
  --mode random \
  --batch_size 2 --learning_rate 1e-2 \
  --train_list BraTS18/BraTS18_metaTrain.csv \
  --val_list BraTS18/BraTS18_metaVal.csv
```

`run_brats_autopilot.sh` resumes from `last.pth` after a GCP host restart, then runs eval. Paper Table 1 averages (ET / TC / WT **63.74 / 79.15 / 86.90**) are from the **online** 66-case set after postprocess. Numbers from `eval_all_modalities.py` are the **local** 57-case test and will not match that row.

---

## ADNI (Table 2, C / B / I)

Full protocol, why G is dropped, and the clinical centering fix: **[adni_flexmoe/README.md](adni_flexmoe/README.md)**.

Loaders are a port of [UNITES-Lab/Flex-MoE](https://github.com/UNITES-Lab/Flex-MoE) (clinical / biospecimen / UCSFFSX7). Genomic is **off by default**. MetaKD and Flex-MoE do not release the Table 2 SNP matrix or `PTID_splits.json`; a Flex-MoE-style PLINK build locally tops out around 43–49% vs paper G **77.32**. Clinical one-hots also have no 59.65-level 3-class signal (logistic ~45, MetaKD ~50). Locked run: drop G, `--center_inputs`, frozen split **2063 / 442 / 442**.

```bash
# data: adni_raw_flexmoe/{diagnosis,clinical,biospecimen,image}/  (ADNI license)

python run_adni_pipeline.py \
  --data_root adni_raw_flexmoe \
  --split_json adni_flexmoe/PTID_splits.json \
  --allow_generated_split \
  --center_inputs \
  --seeds 0,1,2 \
  --output_csv logs/adni_table2_results.csv
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

Locked weights: `adni_flexmoe/checkpoints/best_adni_model_seed0.pth` (and `seed1`, `seed2`). Diagnosis-onset columns PTCOGBEG, PTADDX, and PTADBEG stay excluded, as in the paper.

---

## AudioVision-MNIST (Table 3)

Data recipe, split, and MFCC steps: **[audiovision/README.md](audiovision/README.md)**. Encoders and MFCCs follow [SMIL](https://github.com/deep-real/SMIL).

```bash
# wavs: data/Audio_MNIST/sound/{0-9}/*.wav   MNIST downloads itself

python -m audiovision.run_experiments
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
