
from core import euclidean_distance, quicksort


class KNNClassifier:
    def __init__(self, k=3):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """Store the labeled training set. KNN is 'lazy' -- no training work
        happens until predict() is called."""
        if len(X) != len(y):
            raise ValueError("X and y must be the same length")
        self.X_train = X
        self.y_train = y

    def predict_one(self, point):
        if not self.X_train:
            raise RuntimeError("Call fit() before predict()")

        # 1. Distance Metric: distance from `point` to every training point
        distances = [
            (euclidean_distance(point, x), label)
            for x, label in zip(self.X_train, self.y_train)
        ]

        # 2. Custom Sorting: rank ascending with our own quicksort
        ranked = quicksort(distances, key=lambda pair: pair[0])

        # Isolate the top K closest neighbors
        k = min(self.k, len(ranked))
        neighbors = ranked[:k]

        # 3. Majority Voting (+ tie-break by nearest-neighbor distance)
        return self._majority_vote(neighbors)

    def predict(self, points):
        """Predict a batch of points; returns a list of labels."""
        return [self.predict_one(p) for p in points]

    @staticmethod
    def _majority_vote(neighbors):
        votes = {}          # label -> count
        closest_dist = {}   # label -> distance of its nearest member

        for dist, label in neighbors:
            votes[label] = votes.get(label, 0) + 1
            if label not in closest_dist or dist < closest_dist[label]:
                closest_dist[label] = dist

        best_label = None
        best_count = -1
        best_dist = float("inf")

        for label, count in votes.items():
            if count > best_count:
                best_label, best_count, best_dist = label, count, closest_dist[label]
            elif count == best_count and closest_dist[label] < best_dist:
                # Tie in vote count -> whichever class has the nearer neighbor wins
                best_label, best_dist = label, closest_dist[label]

        return best_label