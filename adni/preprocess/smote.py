"""
SMOTE (Synthetic Minority Over-sampling Technique) implementation.
Following the MetaKD paper: k=5 nearest neighbors interpolation on minority classes (MCI, AD).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(X, y, k_neighbors=5, random_state=42):
    """
    Applies SMOTE on minority classes to balance the training distribution.
    X: np.ndarray of shape (N, D)
    y: np.ndarray of shape (N,) with integer class labels [0, 1, 2]
    """
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    X_balanced = [X.copy()]
    y_balanced = [y.copy()]

    for cls, count in zip(classes, counts):
        n_synthetic_needed = max_count - count
        if n_synthetic_needed <= 0:
            continue

        X_cls = X[y == cls]
        k = min(k_neighbors, len(X_cls) - 1)
        if k < 1:
            # If too few samples, replicate with tiny noise
            indices = np.random.choice(len(X_cls), size=n_synthetic_needed, replace=True)
            synth_samples = X_cls[indices] + np.random.normal(0, 1e-4, size=(n_synthetic_needed, X.shape[1]))
        else:
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X_cls)
            distances, indices = nbrs.kneighbors(X_cls)

            synth_samples = []
            for _ in range(n_synthetic_needed):
                i = np.random.randint(0, len(X_cls))
                # Choose random neighbor from k nearest neighbors (excluding self at index 0)
                neighbor_idx = indices[i, np.random.randint(1, k + 1)]
                diff = X_cls[neighbor_idx] - X_cls[i]
                gap = np.random.rand()
                synth = X_cls[i] + gap * diff
                synth_samples.append(synth)
            synth_samples = np.array(synth_samples)

        X_balanced.append(synth_samples)
        y_balanced.append(np.full(n_synthetic_needed, cls, dtype=y.dtype))

    X_resampled = np.vstack(X_balanced)
    y_resampled = np.concatenate(y_balanced)

    # Shuffle resampled dataset
    perm = np.random.permutation(len(y_resampled))
    return X_resampled[perm], y_resampled[perm]
