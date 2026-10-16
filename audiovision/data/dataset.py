"""Audiovision-MNIST pairs using the SMIL preprocessing path.

MFCC follows deep-real/SMIL ``utils/wav2mfcc.py``: load at the native
rate, keep every third sample, then 20 MFCCs at sr=8000 padded to 20
frames. Images and MFCCs are normalized with mean 0.5 and std 0.5, the
same ``ToTensor`` + ``Normalize([0.5], [0.5])`` transform SMIL applies.

The 1,500-pair pool follows MetaKD: 150 samples per digit. Test files
are SMIL's held-out utterances, index 0-14 (45 per digit, 450 total).
The remaining 105 recordings per digit are shuffled into 90 train and
15 meta-validation. Audio files are balanced across Jackson, Nicolas,
and the remaining speaker, which is how SMIL builds ``SoundMNIST``.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

_AV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_AV_ROOT, ".."))


def _first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def _default_audio_dir():
    return _first_existing(
        os.path.join(_AV_ROOT, "data", "Audio_MNIST", "sound"),
        os.path.join(_REPO_ROOT, "data", "Audio_MNIST", "sound"),
    )


def _default_mnist_dir():
    return _first_existing(
        os.path.join(_AV_ROOT, "data", "MNIST"),
        os.path.join(_REPO_ROOT, "data", "MNIST"),
    )


def _default_cache_path():
    return _first_existing(
        os.path.join(_AV_ROOT, "data", "Audio_MNIST", "audiovision_1500_smil_heldout.npz"),
        os.path.join(_REPO_ROOT, "data", "Audio_MNIST", "audiovision_1500_smil_heldout.npz"),
        os.path.join(_AV_ROOT, "data", "Audio_MNIST", "audiovision_1500_smil_heldout.npz"),
    )


def wav2mfcc(wav_path: str, n_mfcc: int = 20, target_frames: int = 20) -> np.ndarray:
    """SMIL MFCC. Requires librosa, matching ``src/utils/wav2mfcc.py``."""
    import librosa

    wave, _sr = librosa.load(wav_path, mono=True, sr=None)
    wave = np.asfortranarray(wave[::3])
    mfcc = librosa.feature.mfcc(y=wave, sr=8000, n_mfcc=n_mfcc)
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc.astype(np.float32)


def _smil_normalize(array: np.ndarray) -> np.ndarray:
    """Match torchvision Normalize(mean=0.5, std=0.5) on a float array."""
    return ((array - 0.5) / 0.5).astype(np.float32)


def _speaker_balanced_wavs(wav_files, per_class_num: int):
    """SMIL ``get_sound_train_list``: equal counts of jackson, nicolas, other."""
    per_speaker = int(per_class_num / 3)
    jackson, nicolas, others = [], [], []
    tagged = False
    for path in wav_files:
        parts = os.path.basename(path).split("_")
        if len(parts) < 2:
            continue
        tagged = True
        speaker = parts[1]
        if speaker == "jackson" and len(jackson) < per_speaker:
            jackson.append(path)
        elif speaker == "nicolas" and len(nicolas) < per_speaker:
            nicolas.append(path)
        elif speaker not in ("jackson", "nicolas") and len(others) < per_speaker:
            others.append(path)
    chosen = jackson + nicolas + others
    if tagged and len(chosen) >= per_class_num:
        return chosen[:per_class_num]
    return list(wav_files[:per_class_num])


def _recording_index(wav_path: str) -> int:
    """Free Spoken Digit names are ``{digit}_{speaker}_{index}.wav``."""
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    return int(stem.split("_")[2])


def build_audiovision_pairs(
    audio_dir: str = None,
    mnist_dir: str = None,
    cache_path: str = None,
    num_samples_per_digit: int = 150,
):
    """Build 150 paired samples per digit and cache the normalized arrays."""
    audio_dir = audio_dir or _default_audio_dir()
    mnist_dir = mnist_dir or _default_mnist_dir()
    cache_path = cache_path or _default_cache_path()
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["images"], data["audios"], data["labels"], data["recording_index"], data["speakers"]

    print("Building AudioVision-MNIST 1,500 paired dataset (SMIL MFCC)...")
    mnist_train = datasets.MNIST(root=mnist_dir, train=True, download=True)
    mnist_by_digit = {digit: [] for digit in range(10)}
    for img, label in mnist_train:
        bucket = mnist_by_digit[label]
        if len(bucket) < num_samples_per_digit:
            bucket.append(_smil_normalize(np.array(img, dtype=np.float32) / 255.0))

    images, audios, labels, recording_index, speakers = [], [], [], [], []
    for digit in range(10):
        digit_dir = os.path.join(audio_dir, str(digit))
        wav_files = sorted(glob.glob(os.path.join(digit_dir, "*.wav")))
        if len(wav_files) < num_samples_per_digit:
            raise FileNotFoundError(
                f"Need at least {num_samples_per_digit} wav files in {digit_dir}, "
                f"found {len(wav_files)}. SMIL's prepared AV-MNIST archive is at "
                "https://github.com/deep-real/SMIL."
            )
        chosen = _speaker_balanced_wavs(wav_files, num_samples_per_digit)
        if len(chosen) < num_samples_per_digit:
            raise FileNotFoundError(
                f"Speaker-balanced selection for digit {digit} returned {len(chosen)} files."
            )
        for index, wav_path in enumerate(chosen):
            mfcc = _smil_normalize(wav2mfcc(wav_path))
            images.append(mnist_by_digit[digit][index])
            audios.append(mfcc)
            labels.append(digit)
            recording_index.append(_recording_index(wav_path))
            speakers.append(os.path.basename(wav_path).split("_")[1])

    images = np.stack(images).astype(np.float32)
    audios = np.stack(audios).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    recording_index = np.asarray(recording_index, dtype=np.int64)
    speakers = np.asarray(speakers)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        images=images,
        audios=audios,
        labels=labels,
        recording_index=recording_index,
        speakers=speakers,
    )
    print(f"Saved {len(labels)} AudioVision-MNIST pairs to {cache_path}")
    return images, audios, labels, recording_index, speakers


class AudioVisionMNIST(Dataset):
    """MNIST image (1, 28, 28) paired with an MFCC (1, 20, 20)."""

    def __init__(
        self,
        images: np.ndarray,
        audios: np.ndarray,
        labels: np.ndarray,
        missing_modality: str = "none",
        available_rate: float = 1.0,
        seed: int = 42,
    ):
        self.images = images
        self.audios = audios
        self.labels = labels
        self.missing_modality = missing_modality
        self.available_rate = available_rate

        self.visual_mask = np.ones(len(labels), dtype=bool)
        self.audio_mask = np.ones(len(labels), dtype=bool)
        if missing_modality in ("audio", "visual") and available_rate < 1.0:
            rng = np.random.RandomState(seed)
            num_available = int(np.round(len(labels) * available_rate))
            available_idx = rng.choice(len(labels), size=num_available, replace=False)
            if missing_modality == "audio":
                self.audio_mask[:] = False
                self.audio_mask[available_idx] = True
            else:
                self.visual_mask[:] = False
                self.visual_mask[available_idx] = True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": torch.from_numpy(self.images[idx]).unsqueeze(0),
            "audio": torch.from_numpy(self.audios[idx]).unsqueeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "visual_mask": torch.tensor(self.visual_mask[idx], dtype=torch.bool),
            "audio_mask": torch.tensor(self.audio_mask[idx], dtype=torch.bool),
        }


def _split_indices(labels, recording_index, seed):
    """Held-out test is utterance index 0-14. The other 105 clips are 90 train and 15 meta-val."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        held_out = digit_indices[recording_index[digit_indices] < 15]
        pool = digit_indices[recording_index[digit_indices] >= 15].copy()
        if len(held_out) != 45 or len(pool) != 105:
            raise RuntimeError(
                f"Digit {digit} has {len(held_out)} held-out and {len(pool)} train-pool "
                "recordings; expected 45 and 105."
            )
        rng.shuffle(pool)
        train_idx.extend(pool[:90])
        val_idx.extend(pool[90:105])
        test_idx.extend(held_out)
    return train_idx, val_idx, test_idx


def get_audiovision_loaders(
    audio_dir: str = None,
    mnist_dir: str = None,
    batch_size: int = 32,
    missing_modality: str = "none",
    available_rate: float = 1.0,
    seed: int = 42,
):
    """Held-out 450 test, 900 train, 150 meta-val. Missing masks apply only to train."""
    images, audios, labels, recording_index, _speakers = build_audiovision_pairs(audio_dir, mnist_dir)
    train_idx, val_idx, test_idx = _split_indices(labels, recording_index, seed)

    def _subset(indices, drop_modality: str, rate: float):
        indices = np.asarray(indices)
        return AudioVisionMNIST(
            images[indices],
            audios[indices],
            labels[indices],
            missing_modality=drop_modality,
            available_rate=rate,
            seed=seed,
        )

    train_set = _subset(train_idx, missing_modality, available_rate)
    val_set = _subset(val_idx, "none", 1.0)
    test_set = _subset(test_idx, "none", 1.0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
