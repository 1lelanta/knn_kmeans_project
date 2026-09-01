"""
demo.py
-------
Runs both deliverables end-to-end and prints a walkthrough of what's
happening at each step, exactly as the "Code Clarity & Walkthrough"
deliverable asks for. Also renders two SVGs so the results can be inspected
visually: knn_result.svg and kmeans_result.svg.

Run with:  python3 demo.py
"""

from core import euclidean_distance, quicksort
from knn import KNNClassifier
from kmeans import KMeans
from dataset import make_labeled_blobs, make_unlabeled_blobs
from plot_svg import scatter_svg


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_knn():
    section("PART 1 -- K-Nearest Neighbors (supervised classification)")

    X, y = make_labeled_blobs()
    class_names = quicksort(list(set(y)))
    print(f"Training set: {len(X)} labeled 2D points across classes "
          f"{class_names}")

    knn = KNNClassifier(k=5)
    knn.fit(X, y)

    # A handful of query points, including one deliberately placed near a
    # cluster boundary to exercise the tie-break / majority-vote logic.
    queries = [
        [2.2, 1.8],   # deep inside class A
        [8.1, 3.3],   # deep inside class B
        [5.3, 8.7],   # deep inside class C
        [5.0, 5.0],   # ambiguous, near the middle of all three
    ]

    print("\nStep-by-step for the first query point (showing the pipeline):")
    q = queries[0]
    distances = [(euclidean_distance(q, x), label) for x, label in zip(X, y)]
    ranked = quicksort(distances, key=lambda pair: pair[0])
    print(f"  Query point: {q}")
    print(f"  1) Distance metric computed to all {len(X)} training points")
    print(f"  2) Custom quicksort ranks them ascending. Closest 5:")
    for dist, label in ranked[:5]:
        print(f"       distance={dist:.3f}  label={label}")
    votes = {}
    for _, label in ranked[:5]:
        votes[label] = votes.get(label, 0) + 1
    print(f"  3) Vote tally among top-5: {votes}  -> majority class: "
          f"{knn.predict_one(q)}")

    print("\nPredictions for all query points:")
    predictions = knn.predict(queries)
    for q, pred in zip(queries, predictions):
        print(f"  {q} -> predicted class: {pred}")

    groups = {label: [] for label in quicksort(list(set(y)))}
    for point, label in zip(X, y):
        groups[label].append(point)
    groups["query"] = queries
    scatter_svg("knn_result.svg", groups,
                "KNN: training classes + query predictions", star_points=None)
    print("\nSaved visualization -> knn_result.svg")


def demo_kmeans():
    section("PART 2 -- K-Means Clustering (unsupervised)")

    X = make_unlabeled_blobs()
    print(f"Unlabeled dataset: {len(X)} 2D points, no ground-truth clusters")

    k = 3
    model = KMeans(k=k, max_iters=100, tol=1e-4, seed=1)
    model.fit(X)

    print(f"\nConverged with k={k} clusters.")
    print("Final centroids:")
    for i, c in enumerate(model.centroids):
        size = model.labels_.count(i)
        print(f"  Cluster {i}: centroid=({c[0]:.2f}, {c[1]:.2f})  "
              f"points={size}")

    groups = {}
    for i in range(k):
        groups[f"cluster {i}"] = [p for p, lab in zip(X, model.labels_) if lab == i]

    scatter_svg("kmeans_result.svg", groups,
                "K-Means: final cluster assignments (stars = centroids)",
                star_points=model.centroids)
    print("\nSaved visualization -> kmeans_result.svg")


def demo_edge_cases():
    section("EDGE CASES (as required by 'Code Clarity & Walkthrough')")

    print("1) KNN tie in majority voting:")
    X = [[0, 0], [0, 1], [10, 10], [10, 11]]
    y = ["A", "B", "C", "D"]
    knn = KNNClassifier(k=4)  # k = all points -> guaranteed 1-1-1-1 tie
    knn.fit(X, y)
    q = [0, 0.4]
    result = knn.predict_one(q)
    print(f"   4 neighbors, 4 different labels -> perfect tie on vote count.")
    print(f"   Tie-break rule: nearest individual neighbor wins.")
    print(f"   Query {q} -> resolved to class '{result}' "
          f"(closest point is [0, 0] which is class A)")

    print("\n2) K-Means empty cluster:")
    # Two points on top of each other + one far outlier, asking for 3
    # clusters forces an empty cluster on some iterations.
    X2 = [[0, 0], [0, 0.01], [50, 50]]
    model = KMeans(k=3, max_iters=50, tol=1e-6, seed=3)
    model.fit(X2)
    print(f"   3 points, k=3, two points nearly identical -> an initial "
          f"assignment can leave a centroid with zero points.")
    print(f"   Reseed rule: empty centroid moves to the worst-fit point.")
    print(f"   Final centroids: {[[round(v, 3) for v in c] for c in model.centroids]}")
    print(f"   Cluster sizes: {[model.labels_.count(i) for i in range(3)]}")


if __name__ == "__main__":
    demo_knn()
    demo_kmeans()
    demo_edge_cases()
    print("\n" + "=" * 60)
    print("Done. See knn_result.svg and kmeans_result.svg for visuals.")
    print("=" * 60)
