
import random

 """A labeled 2D dataset for KNN: three visually separated classes."""
def make_labeled_blobs(seed=42):
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

# generate a random 2D dataset for K-Means: three loose clusters

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
