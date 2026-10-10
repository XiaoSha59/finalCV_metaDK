"""
PyTorch Dataset and DataLoader for ADNI Multi-Modal Classification.
Loads preprocessed arrays from adni_processed.npz.
Supports arbitrary missing modality mask simulation for 15 modality combinations.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


MODALITIES_MAP = {'C': 0, 'G': 1, 'B': 2, 'I': 3}


class ADNIDataset(Dataset):
    def __init__(self, npz_path="adni/data_processed/adni_processed.npz", split="train", active_modalities="C,G,B,I", filter_available=True):
        """
        split: 'train', 'val', or 'test'
        active_modalities: comma-separated string, e.g. 'C,G,B,I', 'C,G', 'C', etc.
        filter_available: if True, filters evaluation samples to those with at least one active modality present.
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Processed dataset not found at {npz_path}. Run data_processor.py first.")

        data = np.load(npz_path)
        self.c = data[f'c_{split}']
        self.g = data[f'g_{split}']
        self.b = data[f'b_{split}']
        self.i = data[f'i_{split}']
        self.masks = data[f'masks_{split}']
        self.y = data[f'y_{split}']
        self.dims = data['dims']

        # Determine which modalities are active for this evaluation
        self.active_indices = [MODALITIES_MAP[m.strip().upper()] for m in active_modalities.split(',') if m.strip().upper() in MODALITIES_MAP]

        # For test/val evaluation of specific subsets, filter to samples that possess at least one active modality
        if split in ['val', 'test'] and filter_available and len(self.active_indices) < 4:
            valid_mask = np.any([self.masks[:, m] for m in self.active_indices], axis=0)
            if np.sum(valid_mask) > 0:
                self.c = self.c[valid_mask]
                self.g = self.g[valid_mask]
                self.b = self.b[valid_mask]
                self.i = self.i[valid_mask]
                self.masks = self.masks[valid_mask]
                self.y = self.y[valid_mask]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_c = torch.tensor(self.c[idx], dtype=torch.float32)
        x_g = torch.tensor(self.g[idx], dtype=torch.float32)
        x_b = torch.tensor(self.b[idx], dtype=torch.float32)
        x_i = torch.tensor(self.i[idx], dtype=torch.float32)

        # Base presence mask
        mask = self.masks[idx].copy()
        # If simulating missing modality at test time, zero out inactive modalities
        for m_idx in range(4):
            if m_idx not in self.active_indices:
                mask[m_idx] = False

        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        return (x_c, x_g, x_b, x_i), mask_tensor, label


def get_adni_dataloaders(npz_path="adni/data_processed/adni_processed.npz", batch_size=32, num_workers=0):
    """Returns train, val, and test DataLoaders."""
    train_ds = ADNIDataset(npz_path=npz_path, split="train", filter_available=False)
    val_ds = ADNIDataset(npz_path=npz_path, split="val", filter_available=False)
    test_ds = ADNIDataset(npz_path=npz_path, split="test", filter_available=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.dims
