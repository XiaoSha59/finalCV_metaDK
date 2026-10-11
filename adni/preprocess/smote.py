"""
SMOTE (Synthetic Minority Over-sampling Technique) implementation.
Following the MetaKD paper: k=5 nearest neighbors interpolation on minority classes (MCI, AD).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(X, y, k_neighbors=5, target_counts=None, target_ratios=None, random_state=42):
    """
    Applies SMOTE (k=5) on minority classes to address severe class imbalance
    while maintaining the natural underlying population distribution characteristics
    (~50-60% CTL, 25-35% MCI, 15-20% AD) as specified in MetaKD paper Section 8.
    
    Avoids artificial flat 1:1:1 oversampling which distorts population priors
    and collapses test accuracy on majority classes.
    """
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(classes, counts))

    # Determine number of synthetic samples needed per class
    if target_counts is not None:
        needed_dict = {cls: max(0, target_counts.get(cls, count) - count) for cls, count in count_dict.items()}
    elif target_ratios is not None:
        if isinstance(target_ratios, (list, tuple)):
            target_ratios = {i: r for i, r in enumerate(target_ratios)}
        max_cls = max(count_dict, key=count_dict.get)
        base_n = count_dict[max_cls] / target_ratios[max_cls]
        needed_dict = {cls: max(0, int(round(base_n * target_ratios[cls])) - count_dict[cls]) for cls in classes}
    else:
        # Default proportional SMOTE: moderately augment severe minority class (AD) by +30%
        # to enhance sensitivity while preserving natural distribution characteristics
        needed_dict = {}
        for cls, count in count_dict.items():
            if count < counts[0] * 0.5:  # severe minority class (AD)
                needed_dict[cls] = int(round(count * 0.30))
            else:
                needed_dict[cls] = 0

    X_balanced = [X.copy()]
    y_balanced = [y.copy()]

    for cls in classes:
        n_synthetic_needed = needed_dict.get(cls, 0)
        if n_synthetic_needed <= 0:
            continue

        X_cls = X[y == cls]
        k = min(k_neighbors, len(X_cls) - 1)
        if k < 1:
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

