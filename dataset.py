"""
dataset.py
----------
Small synthetic 2D datasets, generated with only the standard `random`
module (no numpy). The slides reference "the provided 2D dataset" for KNN;
since no dataset file came with the deck, this generates one deterministically
(fixed seed) so results are reproducible.
"""

import random


def make_labeled_blobs(seed=42):
    """A labeled 2D dataset for KNN: three visually separated classes."""
    rng = random.Random(seed)
    centers = {
        "A": (2.0, 2.0),
        "B": (8.0, 3.0),
        "C": (5.0, 9.0),
    }
    X, y = [], []
    for label, (cx, cy) in centers.items():
        for _ in range(15):
            x = cx + rng.uniform(-1.2, 1.2)
            yy = cy + rng.uniform(-1.2, 1.2)
            X.append([x, yy])
            y.append(label)
    return X, y


def make_unlabeled_blobs(seed=7):
    """An unlabeled 2D dataset for K-Means: three loose clusters."""
    rng = random.Random(seed)
    centers = [(2.0, 3.0), (9.0, 8.0), (7.0, 1.0)]
    X = []
    for cx, cy in centers:
        for _ in range(20):
            x = cx + rng.uniform(-1.5, 1.5)
            yy = cy + rng.uniform(-1.5, 1.5)
            X.append([x, yy])
    rng.shuffle(X)
    return X
