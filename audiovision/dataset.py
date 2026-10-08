import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fftpack
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from PIL import Image


def extract_mfcc(wav_path: str, n_mfcc: int = 20, target_frames: int = 20) -> np.ndarray:
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from a .wav audio file
    using standard scipy signal processing without numba/librosa dependencies.
    Output shape: (n_mfcc, target_frames) = (20, 20).
    """
    sr, y = wav.read(wav_path)
    if len(y.shape) > 1:
        y = y[:, 0]
    if y.dtype != np.float32:
        y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-8)
    
    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # Framing
    frame_len = max(int(sr * 0.025), 1)
    frame_step = max(int(sr * 0.010), 1)
    signal_len = len(y)
    num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step)) + 1
    
    pad_signal_len = num_frames * frame_step + frame_len
    z = np.zeros((pad_signal_len - signal_len))
    pad_signal = np.append(y, z)
    
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_len)
    
    # FFT & Power spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    
    # Mel Filterbank
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mfcc + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin_points = np.floor((NFFT + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((n_mfcc, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_mfcc + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1] + 1e-8)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m] + 1e-8)
            
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    # DCT
    mfcc = fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    mfcc = mfcc.T  # (n_mfcc, frames)
    
    # Resize / pad to target_frames (20, 20)
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]
        
    mean = np.mean(mfcc)
    std = np.std(mfcc) + 1e-8
    return ((mfcc - mean) / std).astype(np.float32)


def build_audiovision_pairs(
    audio_dir: str = "data/Audio_MNIST/sound",
    mnist_dir: str = "data/MNIST",
    cache_path: str = "data/Audio_MNIST/audiovision_1500_pairs.npz",
    num_samples_per_digit: int = 150
):
    """
    Builds and caches the 1,500 paired Audio-Visual MNIST dataset.
    150 samples per digit (0-9) = 1,500 total samples.
    """
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['images'], data['audios'], data['labels']

    print("Building AudioVision-MNIST 1,500 paired dataset...")
    
    # Load MNIST
    mnist_train = datasets.MNIST(root=mnist_dir, train=True, download=True)
    mnist_by_digit = {d: [] for d in range(10)}
    for img, label in mnist_train:
        if len(mnist_by_digit[label]) < num_samples_per_digit:
            img_arr = np.array(img, dtype=np.float32) / 255.0  # (28, 28)
            img_arr = (img_arr - 0.1307) / 0.3081  # Standard MNIST normalization
            mnist_by_digit[label].append(img_arr)
            
    images = []
    audios = []
    labels = []

    for digit in range(10):
        digit_dir = os.path.join(audio_dir, str(digit))
        wav_files = sorted(glob.glob(os.path.join(digit_dir, "*.wav")))[:num_samples_per_digit]
        
        for i, wav_f in enumerate(wav_files):
            mfcc = extract_mfcc(wav_f, n_mfcc=20, target_frames=20)
            img = mnist_by_digit[digit][i]
            
            images.append(img)
            audios.append(mfcc)
            labels.append(digit)

    images = np.array(images, dtype=np.float32)  # (1500, 28, 28)
    audios = np.array(audios, dtype=np.float32)  # (1500, 20, 20)
    labels = np.array(labels, dtype=np.int64)    # (1500,)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, images=images, audios=audios, labels=labels)
    print(f"Saved {len(labels)} AudioVision-MNIST pairs to {cache_path}")
    return images, audios, labels


class AudioVisionMNIST(Dataset):
    """
    AudioVision-MNIST Multi-Modal Dataset.
    Combines MNIST images (28x28x1) and FSDD audio MFCCs (20x20x1).
    """
    def __init__(
        self,
        images: np.ndarray,
        audios: np.ndarray,
        labels: np.ndarray,
        missing_modality: str = "none",  # "none", "audio", "visual"
        available_rate: float = 1.0,     # Rate of available modality in training (e.g. 0.05, 0.10, 0.15, 0.20)
        seed: int = 42
    ):
        self.images = images
        self.audios = audios
        self.labels = labels
        self.missing_modality = missing_modality
        self.available_rate = available_rate

        self.visual_mask = np.ones(len(labels), dtype=bool)
        self.audio_mask = np.ones(len(labels), dtype=bool)

        # Apply missing rate mask if specified
        if missing_modality == "audio" and available_rate < 1.0:
            np.random.seed(seed)
            num_available = int(np.round(len(labels) * available_rate))
            available_idx = np.random.choice(len(labels), size=num_available, replace=False)
            self.audio_mask[:] = False
            self.audio_mask[available_idx] = True

        elif missing_modality == "visual" and available_rate < 1.0:
            np.random.seed(seed)
            num_available = int(np.round(len(labels) * available_rate))
            available_idx = np.random.choice(len(labels), size=num_available, replace=False)
            self.visual_mask[:] = False
            self.visual_mask[available_idx] = True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Image shape: (1, 28, 28)
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)
        # Audio shape: (1, 20, 20)
        aud = torch.from_numpy(self.audios[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        v_mask = torch.tensor(self.visual_mask[idx], dtype=torch.bool)
        a_mask = torch.tensor(self.audio_mask[idx], dtype=torch.bool)

        return {
            'image': img,
            'audio': aud,
            'label': label,
            'visual_mask': v_mask,
            'audio_mask': a_mask
        }


def get_audiovision_loaders(
    audio_dir: str = "data/Audio_MNIST/sound",
    mnist_dir: str = "data/MNIST",
    batch_size: int = 32,
    missing_modality: str = "none",
    available_rate: float = 1.0,
    seed: int = 42
):
    """
    Creates stratified Train (60% = 900), Meta-Val (10% = 150), and Test (30% = 450) DataLoaders.
    """
    images, audios, labels = build_audiovision_pairs(audio_dir, mnist_dir)
    
    np.random.seed(seed)
    # Stratified 60 / 10 / 30 split per digit
    train_idx, val_idx, test_idx = [], [], []
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        np.random.shuffle(digit_indices)
        
        # 150 samples -> 90 train (60%), 15 val (10%), 45 test (30%)
        train_idx.extend(digit_indices[:90])
        val_idx.extend(digit_indices[90:105])
        test_idx.extend(digit_indices[105:150])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    train_set = AudioVisionMNIST(
        images[train_idx], audios[train_idx], labels[train_idx],
        missing_modality=missing_modality, available_rate=available_rate, seed=seed
    )
    val_set = AudioVisionMNIST(
        images[val_idx], audios[val_idx], labels[val_idx],
        missing_modality="none", available_rate=1.0, seed=seed
    )
    test_set = AudioVisionMNIST(
        images[test_idx], audios[test_idx], labels[test_idx],
        missing_modality="none", available_rate=1.0, seed=seed
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
