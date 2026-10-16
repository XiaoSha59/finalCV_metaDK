# AudioVision-MNIST

Image + spoken-digit pairs used for MetaKD Table 3 ([arXiv:2405.07155](https://arxiv.org/abs/2405.07155)). Encoders and the MFCC recipe follow [SMIL](https://github.com/deep-real/SMIL). The official MetaKD repo ([billhhh/MetaKD](https://github.com/billhhh/MetaKD)) has BraTS only, so this folder is the AudioVision run.

## Data

1,500 pairs, 150 per digit `0–9`.

| Modality | Source | Tensor |
|---|---|---|
| Image | First 150 MNIST training images of that digit ([torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)) | `1 × 28 × 28` |
| Audio | [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset), 50 clips each from jackson, nicolas, and theo | `1 × 20 × 20` MFCC |

### Data Acquisition & Placement
- **Images (MNIST)**: Downloaded automatically into `audiovision/data/MNIST/` via `torchvision.datasets.MNIST`.
- **Audio (FSDD)**: Clone or download [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset):
  ```bash
  git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
  ```

Put the audio recordings under `audiovision/data/Audio_MNIST/sound/` (gitignored), grouped by digit `0-9`:

```text
audiovision/data/Audio_MNIST/sound/0/*.wav
audiovision/data/Audio_MNIST/sound/1/*.wav
...
audiovision/data/Audio_MNIST/sound/9/*.wav
```

A leftover copy at repo-root `data/Audio_MNIST/` is still accepted by the loader. On first run, preprocessed arrays are cached at `audiovision/data/Audio_MNIST/audiovision_1500_smil_heldout.npz`.

## Processing

Audio follows SMIL `src/utils/wav2mfcc.py`:

1. Load the wav in mono at its native sample rate.
2. Keep every third sample (`wave[::3]`).
3. Compute 20 MFCCs with `sr=8000`.
4. Pad or truncate to 20 frames.

Images are divided by 255. Both images and MFCCs are then normalized with mean `0.5` and std `0.5`. No Gaussian noise is added.

The arrays are cached at `audiovision/data/Audio_MNIST/audiovision_1500_smil_heldout.npz`.

## Split

Seed `42`. Per digit:

- **Test (450):** utterance index `0–14` (15 clips × 3 speakers = 45).
- **Train (900) and meta-val (150):** index `15–49` (105 clips) is shuffled, then cut into 90 train and 15 validation.

A missing modality is applied only to the training set. `available_rate` is the fraction that **keeps** that modality. Validation and test stay complete; the test mode drops one modality in the model.

- Table 3a: hide audio at 5%, 10%, 15%, 20% during training. Test with images only.
- Table 3b: hide images at those rates. Test with audio only.

## Reproduce

```bash
pip install -r requirements.txt
python -m audiovision.process.run_experiments
```

Training is 60 epochs, Adam, batch size 32, seed 42. Checkpoints from the locked run are already in `audiovision/checkpoints/`.

| | 5% | 10% | 15% | 20% | Lower bound | Upper bound |
|---|---:|---:|---:|---:|---:|---:|
| Table 3a, image test | 96.22 | 94.67 | 94.44 | 96.22 | 95.33 | 96.67 |
| Table 3b, audio test | 85.78 | 88.22 | 87.78 | 87.56 | 90.67 | 96.67 |

Paper Table 3 reports MetaKD at 93.56–95.11 (3a) and 84.67–88.44 (3b). Its audio lower bound is 80.44, which is SMIL's 21-clip checkpoint, not this 900-clip training set.

## References

- MetaKD paper: https://arxiv.org/abs/2405.07155
- MetaKD code (BraTS only): https://github.com/billhhh/MetaKD
- SMIL code and MFCC: https://github.com/deep-real/SMIL
- SMIL paper: https://arxiv.org/abs/2103.05677
- Free Spoken Digit Dataset: https://github.com/Jakobovski/free-spoken-digit-dataset
