"""
kmeans.py
---------
K-Means clustering, implemented per the "Deep Dive: K-Means Mechanics" spec:

  1. Centroid Init   -> K starting centers chosen from the data points
  2. Assignment Step -> every point assigned to its nearest centroid
  3. Update Step     -> each centroid recomputed as the mean of its cluster

Loops until convergence (centroids stop moving beyond `tol`) or `max_iters`
is reached.

Edge case handled: empty clusters. If an assignment step leaves some
centroid with zero points, the mean update is undefined (0/0). Instead of
crashing or silently freezing that centroid, it is reseeded at the data
point that is currently *farthest* from its own assigned centroid -- the
point the model is fitting worst -- which gives the empty cluster a fresh,
useful starting position for the next iteration.
"""

import random

from core import euclidean_distance


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, seed=None):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.rng = random.Random(seed)
        self.centroids = []
        self.labels_ = []
        self.history_ = []

    def fit(self, X):
        if len(X) < self.k:
            raise ValueError("Need at least k points to form k clusters")

        # 1. Centroid Init: pick k distinct data points as starting centers
        self.centroids = [list(p) for p in self.rng.sample(X, self.k)]
        self.history_ = []

        for iteration in range(self.max_iters):
            # 2. Assignment Step
            labels = [self._nearest_centroid(p) for p in X]

            # 3. Update Step (with empty-cluster handling)
            new_centroids = self._update_centroids(X, labels)

            # Convergence check: how far did each centroid move?
            shift = max(
                euclidean_distance(old, new)
                for old, new in zip(self.centroids, new_centroids)
            )
            self.centroids = new_centroids
            self.labels_ = labels
            self.history_.append({
                "centroids": [list(c) for c in self.centroids],
                "labels": list(self.labels_),
            })

            if shift < self.tol:
                break

        return self

    def predict(self, X):
        return [self._nearest_centroid(p) for p in X]

    def _nearest_centroid(self, point):
        best_index = 0
        best_dist = float("inf")
        for i, c in enumerate(self.centroids):
            d = euclidean_distance(point, c)
            if d < best_dist:
                best_dist, best_index = d, i
        return best_index

    def _update_centroids(self, X, labels):
        dims = len(X[0])
        sums = [[0.0] * dims for _ in range(self.k)]
        counts = [0] * self.k

        for point, cluster_id in zip(X, labels):
            counts[cluster_id] += 1
            for d in range(dims):
                sums[cluster_id][d] += point[d]

        new_centroids = []
        for cluster_id in range(self.k):
            if counts[cluster_id] == 0:
                # Empty cluster: reseed at the point currently worst-served
                # by its own centroid (farthest from the centroid it was
                # assigned to), so the empty slot moves somewhere useful.
                new_centroids.append(self._farthest_point(X, labels))
            else:
                mean = [s / counts[cluster_id] for s in sums[cluster_id]]
                new_centroids.append(mean)

        return new_centroids

    def _farthest_point(self, X, labels):
        farthest = None
        farthest_dist = -1.0
        for point, cluster_id in zip(X, labels):
            d = euclidean_distance(point, self.centroids[cluster_id])
            if d > farthest_dist:
                farthest_dist, farthest = d, point
        return list(farthest)
