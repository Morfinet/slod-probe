from collections import defaultdict

import numpy as np

from utils import LABELS


def balanced_indices(labels, seed):
    """Downsample classes to the size of the smallest class."""
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)

    if any(label not in groups for label in LABELS):
        raise ValueError("One of the classes is missing")

    rng = np.random.default_rng(seed)
    size = min(len(groups[label]) for label in LABELS)
    selected = []
    for label in LABELS:
        selected.extend(rng.choice(groups[label], size, replace=False))
    return np.asarray(sorted(selected), dtype=int)
