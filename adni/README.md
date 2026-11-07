# ADNI (Clinical, Genomic, Biospecimen, Imaging)

CN / MCI / AD 3-class staging benchmark for MetaKD Table 2 ([arXiv:2405.07155](https://arxiv.org/abs/2405.07155)). Tabular loaders follow [UNITES-Lab/Flex-MoE](https://github.com/UNITES-Lab/Flex-MoE). The repository supports both:
1. **Full 4-Modality Benchmark (C, G, B, I)**: All 15 missing-modality scenarios across clinical, genomic, biospecimen, and MRI imaging features.
2. **Clean 3-Modality Benchmark (C, B, I)**: 7 non-genomic scenarios with locked 3-seed checkpoints.

## Data

### Data Acquisition & License
- **Official Source**: Alzheimer's Disease Neuroimaging Initiative ([ADNI](https://adni.loni.usc.edu)). Access requires submitting a Data Use Agreement (DUA).
- **Tabular Preprocessed Package**: Follows the curated multi-modal table dump from [UNITES-Lab/Flex-MoE](https://github.com/UNITES-Lab/Flex-MoE).

Place the Flex-MoE ADNI package under `adni/data/raw/` (gitignored). A leftover copy at repo-root `adni_raw_flexmoe/` is still accepted.

```text
adni/data/raw/
├── diagnosis/DXSUM.csv
├── clinical/{MEDHIST,NEUROEXM,PTDEMOG,RECCMEDS,VITALS}.rda
├── biospecimen/APOERES.csv
├── biospecimen/UPENNBIOMK_ROCHE_ELECSYS.rda
└── image/UCSFFSX7.csv
```

| Modality | Source | Encoding | Width here |
|---|---|---|---:|
| C | ADNIMERGE2 `MEDHIST`, `NEUROEXM`, `PTDEMOG`, `RECCMEDS`, `VITALS` | latest `UPDATE_STAMP`, Flex-MoE numeric / one-hot / 5-bin dates, MinMax `[-1, 1]` | 4439 |
| B | Roche Elecsys CSF (`Aβ40`, `Aβ42`, Tau, pTau) + `APOERES` | same Flex-MoE table encoder | 190 |
| I | UCSF FreeSurfer 7.x `UCSFFSX7` (`ST*` `CV` / `TA` / `SV`) | latest stamp, `StandardScaler` | 187 |

Labels: subjects whose `DXSUM.DIAGNOSIS` is unique and in `{1,2,3}` → CN / MCI / AD. On this dump: 2947 subjects (1383 / 1017 / 547). A row is “observed” for a modality if that subject appears in the encoded table; unobserved entries are filled with 0, matching Flex-MoE `reindex`.

## Split

Flex-MoE uses 70 / 15 / 15 with **val and test drawn from the intersection of the active modalities**. Their `PTID_splits.json` is not public.

The locked split is `adni/data/PTID_splits.json` (seed 42, same protocol, C ∩ B ∩ I for val/test): **2063 / 442 / 442**. Do not regenerate it. Pass `--allow_generated_split` so the loader may *read* this file when Flex-MoE’s official JSON is absent; an existing file is never overwritten.

Test majority-class accuracy is **41.86%**.

## Preprocessing

Port of Flex-MoE, not a re-design:

1. Per table, sort by `UPDATE_STAMP`, forward/back fill within `PTID`, keep the last row.
2. Outer-merge on `PTID`, then drop columns that appear in every table (`RID`, `SITEID`, `ORIGPROT`, `VISDATE`, …).
3. Drop `PTCOGBEG`, `PTADDX`, `PTADBEG` (MetaKD and Flex-MoE both exclude these diagnosis-onset fields).
4. Numerics: strip `<>`, MinMax to `[-1, 1]`, mean fill. Binary / string columns with `< 1000` unique values: mode fill, one-hot, `{True, False} → {1, -1}`. Date columns: 5 uniform bins.
5. Imaging is the exception: `StandardScaler` on FreeSurfer volumes and thicknesses, as in Flex-MoE’s preprocessed-image path.

SMOTE (`k = 5`) is applied only on **complete** training rows so missingness is not interpolated, then the class counts are balanced to the majority.

## MetaKD

Algorithm 1, two stages per batch: update w* on a meta-val split D_v, then train theta, zeta with w* frozen. Classification hyperparameters follow the AudioVision supplement (the paper does not give an ADNI optimizer).

| | Value |
|---|---|
| Inner optimizer | Adam 1e-3, weight decay 1e-2, StepLR every 20 epochs x 0.9 |
| IWV optimizer | Adam 1e-2, weight decay 1e-2, one step per train batch |
| D_v | 10% of the training split, stratified, disjoint from D_t |
| Encoder | MLP d -> 256 -> 128 per modality (C/B/I are all < 8192) |
| Decoder (Eq. 3) | concat -> Linear 64, dropout 0.6, Linear 3 |
| Missing slots (Eq. 2) | mean of the observed f_i in the batch |
| KD (Eq. 5) | p = 1, ratio w_i* / w_j*, **mean** L1 (vector-sum L1 with alpha = 0.1 collapses every scenario to the majority class) |
| alpha | 0.1 |
| Batch / epochs | 8 / 60 |
| Missing-modality train | drop 0 to n-1 of the *observed* modalities (BraTS non-dedicated setting in supplement Sec. 8) |
| Checkpoint | best mean accuracy over the 7 C/B/I Table 2 subsets on val |

`115k` / `120k` iterations in the paper are BraTS (and the Fig. 9 IWV plot), not an ADNI recipe. AudioVision is 60 epochs; that is what this run uses.

## Why genomic is omitted

Paper Table 2 reports G-only accuracy **77.32**. MetaKD does not publish SNP QC, liftOver, LD, or the subject list. Flex-MoE describes five PLINK cohorts, hg38, and `plink --indep-pairwise 50 5 0.1`, but not the exact matrix behind MetaKD’s number.

We built that Flex-MoE genomic pipeline locally (five cohorts, liftOver, LD 50 5 0.1). It does not reproduce 77.32:

| Probe | Test acc |
|---|---:|
| Majority class | 41.86 |
| Logistic on the SNP matrix | ~43–44 |
| Logistic on chip / cohort one-hots | ~49 |
| MetaKD, G-only (official-style PLINK) | 26–34 |
| Paper G | 77.32 |

The intersection matrix also lacks the APOE tagging SNPs `rs429358` / `rs7412`. Extra iterations, PCA, or SMOTE-without-G do not create that signal. For baseline MetaKD reproduction, `--drop_g` is the default to isolate clean tabular signals. To evaluate full 4-modality with genomic leakage analysis and CDU-KD across all 15 scenarios, see `prove_c_and_g_leakage.py` and `improvements/adni_uncertainty/`.

## Clinical (C)

Paper C is **59.65**. Two issues are specific to this table, and they are not the same as G.

**Collapse.** Flex-MoE MinMax `[-1, 1]` plus thousands of sparse one-hots (mostly `RECCMEDS` / `PTDEMOG`) leaves a large DC offset. Without correction, C-only MetaKD predicts the majority class (41.86). `--center_inputs` subtracts the **train-observed** column mean before the encoders. That is not in Flex-MoE; it is required for C to train at all. The locked run always passes it.

**Ceiling.** After centering, C is still not 59.65. The same Flex-MoE C matrix, frozen split, no MetaKD:

| Features | Acc |
|---|---:|
| Pipeline C (4439-d, logistic / RF / MLP) | 44.6 / 45.0 / 46.6 |
| Age / sex / education only | 45.7 |
| True vitals (weight, height, BP) | ~40 |
| MetaKD C, 3 seeds, centered | ~50.5 |
| Paper C | 59.65 |
| `PTCOGBEG` + `PTADDX` + `PTADBEG` only | 87.8 |

So C is a **feature** limit, not an under-trained MetaKD branch. Logistic B (~57.5) and I (~54.5–56.6) already sit on the paper’s B / I numbers. Putting C back on B+I *lowers* a linear fusion. We do not put diagnosis-onset dates or site / protocol IDs back in to chase 59.65.

## Official Table 2 Benchmark Results: MetaKD Reproduce vs. Paper Table 2

Evaluation across all 15 combinations of the 4 modalities (Clinical, Genomic, Biospecimen, MRI Image) on the frozen test split (442 subjects, 3 seeds):

| Scenario | Observed Modalities | MetaKD Repro Acc (%) | MetaKD Repro F1 (%) | Paper Acc (%) | Paper F1 (%) |
|:---|:---|:---:|:---:|:---:|:---:|
| C | Clinical only | 49.25 | 47.58 | 59.65 | 43.25 |
| G | Genomic only | 33.71 | 26.39 | 77.32* | 47.32 |
| B | Biospecimen only | 52.19 | 50.79 | 56.23 | 40.57 |
| I | MRI only | 57.01 | 57.41 | 55.32 | 39.05 |
| C, G | Clinical + Genomic | 48.42 | 46.99 | 61.35 | 42.36 |
| C, B | Clinical + Biospecimen | 53.09 | 52.15 | 63.24 | 45.87 |
| C, I | Clinical + MRI | 60.41 | 60.64 | 62.75 | 41.23 |
| G, B | Genomic + Biospecimen | 45.55 | 44.72 | 63.76 | 38.63 |
| G, I | Genomic + MRI | 50.68 | 52.04 | 64.87 | 44.21 |
| B, I | Biospecimen + MRI | 60.03 | 60.67 | 57.56 | 39.54 |
| C, G, B | Clinical + Genomic + Biospecimen | 52.19 | 50.98 | 60.35 | 42.11 |
| C, G, I | Clinical + Genomic + MRI | 56.49 | 56.26 | 65.04 | 50.43 |
| C, B, I | Clinical + Biospecimen + MRI | 61.54 | 62.03 | 63.06 | 42.65 |
| G, B, I | Genomic + Biospecimen + MRI | 57.54 | 58.53 | 64.12 | 55.09 |
| C, G, B, I | Full (All 4 Modalities) | 59.80 | 59.63 | 67.87 | 57.23 |
| **Mean** | **Overall 15-Combination Average** | **53.19** | **52.45** | **62.83*** | **44.64** |

*Note: Genomic accuracy for Paper MetaKD (77.32%) reflects data leakage from unpruned diagnosis-onset variables. Our reproduced baseline on raw SNPs obtains 33.71% Acc (26.39% F1). Across all 15 scenarios, MetaKD Reproduce achieves 53.19% Acc and 52.45% Macro-F1 (+7.81% F1 over Paper 44.64%). For our proposed Class-Decoupled Uncertainty extension (CDU-KD, achieving 54.56% Acc and protecting transitional MCI sensitivity up to 88.6%), see [improvements/adni_uncertainty/README.md](../improvements/adni_uncertainty/README.md).*

Best-validation weights: 3-modality baseline in `adni/checkpoints/reproduce_3mod/` (and root `adni/checkpoints/best_adni_model_seed{0,1,2}.pth`) and full 4-modality baseline in `adni/checkpoints/reproduce_4mod/best_adni_model_seed{0,1,2}.pth`.

## Reproduce

```bash
pip install -r requirements.txt

python -m adni.process.run_pipeline \
 --allow_generated_split \
 --center_inputs \
 --epochs 60 \
 --batch_size 8 \
 --seeds 0,1,2 \
 --output_csv logs/adni_table2_results.csv
```

`--drop_g` is already on. `--no_class_weight` is already the default. CPU is enough for this MLP; a GPU only speeds the 60-epoch loop.

Sanity-check loaders and the frozen split without training:

```bash
python -m adni.process.run_pipeline \
 --allow_generated_split \
 --center_inputs \
 --dry_run
```

Expect `C: 4439`, `B: 190`, `I: 187`, split `2063 / 442 / 442`.

## References

- MetaKD paper: https://arxiv.org/abs/2405.07155
- MetaKD code (BraTS only): https://github.com/billhhh/MetaKD
- Flex-MoE paper: https://arxiv.org/abs/2410.08245
- Flex-MoE code and ADNI tables: https://github.com/UNITES-Lab/Flex-MoE
- ADNI: https://adni.loni.usc.edu
