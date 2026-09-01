"""
test_project.py
----------------
Basic correctness checks for the from-scratch implementations.
Run with:  python3 -m pytest test_project.py -v
        or: python3 test_project.py
"""

import math

from core import euclidean_distance, quicksort
from knn import KNNClassifier
from kmeans import KMeans


def test_euclidean_distance():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0  # classic 3-4-5 triangle
    assert math.isclose(euclidean_distance([1, 1], [1, 1]), 0.0)
    assert math.isclose(euclidean_distance([0, 0, 0], [1, 1, 1]), math.sqrt(3))


def test_quicksort_matches_expected_order():
    data = [5, 3, 8, 1, 9, 2]
    assert quicksort(data) == [1, 2, 3, 5, 8, 9]
    assert data == [5, 3, 8, 1, 9, 2]  # original untouched

    tagged = [(3, "z"), (1, "a"), (2, "b")]
    ranked = quicksort(tagged, key=lambda t: t[0])
    assert [t[1] for t in ranked] == ["a", "b", "z"]


def test_quicksort_edge_cases():
    assert quicksort([]) == []
    assert quicksort([1]) == [1]
    assert quicksort([2, 2, 1, 1]) == [1, 1, 2, 2]


def test_knn_predicts_obvious_neighbor():
    X = [[0, 0], [0, 1], [10, 10], [10, 11], [10, 9]]
    y = ["near", "near", "far", "far", "far"]
    knn = KNNClassifier(k=1)
    knn.fit(X, y)
    assert knn.predict_one([0.2, 0.2]) == "near"
    assert knn.predict_one([10.2, 10.2]) == "far"


def test_knn_majority_vote():
    X = [[0, 0], [0, 1], [0, 2], [5, 5]]
    y = ["A", "A", "B", "B"]
    knn = KNNClassifier(k=3)
    knn.fit(X, y)
    # 3 nearest to [0,0.5] are the three points at (0,0),(0,1),(0,2) -> 2 A vs 1 B
    assert knn.predict_one([0, 0.5]) == "A"


def test_knn_tie_break_uses_nearest_neighbor():
    X = [[0, 0], [10, 10]]
    y = ["A", "B"]
    knn = KNNClassifier(k=2)  # forces a 1-1 tie every time
    knn.fit(X, y)
    assert knn.predict_one([0.1, 0.1]) == "A"   # closer to A
    assert knn.predict_one([9.9, 9.9]) == "B"   # closer to B


def test_kmeans_separates_obvious_clusters():
    X = [[0, 0], [0, 1], [1, 0], [1, 1],       # cluster near (0.5, 0.5)
         [20, 20], [20, 21], [21, 20], [21, 21]]  # cluster near (20.5, 20.5)
    model = KMeans(k=2, seed=0)
    model.fit(X)
    # every point in the first group should share a label, distinct from the second group
    labels = model.labels_
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_kmeans_handles_empty_cluster_without_crashing():
    X = [[0, 0], [0, 0.01], [50, 50]]
    model = KMeans(k=3, seed=3)
    model.fit(X)  # must not raise, even though a centroid can start empty
    assert len(model.centroids) == 3
    assert sum(model.labels_.count(i) for i in range(3)) == len(X)


def test_kmeans_converges_to_stable_centroids():
    X = [[0, 0], [0, 2], [2, 0], [2, 2],
         [10, 10], [10, 12], [12, 10], [12, 12]]
    model = KMeans(k=2, seed=5, tol=1e-6)
    model.fit(X)
    centroid_sums = sorted(round(sum(c), 2) for c in model.centroids)
    assert centroid_sums == [2.0, 22.0]  # (1,1) and (11,11) each sum to 2 / 22


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS: {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed.")
