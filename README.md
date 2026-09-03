# Hands-On AI From Scratch: KNN & K-Means

Bare-bones implementations of K-Nearest Neighbors and K-Means, built to the
spec in the training deck: **zero external ML libraries** — no `numpy`,
`pandas`, `sklearn`, and no built-in sort/distance helpers standing in for
the algorithmic core. Only the standard `math` (for `sqrt`) and `random`
(for centroid init / dataset generation) modules are used.

## Files

| File | Purpose |
|---|---|
| `core.py` | `euclidean_distance()` and a hand-written `quicksort()` — the two primitives both algorithms are built on |
| `knn.py` | `KNNClassifier`: distance → custom sort → majority vote |
| `kmeans.py` | `KMeans`: centroid init → assignment step → update step, looped to convergence |
| `dataset.py` | Synthetic 2D datasets (deterministic, seeded) — the deck references "the provided 2D dataset" but none shipped with it, so this generates one |
| `plot_svg.py` | Dependency-free SVG scatter-plot renderer and self-contained K-Means animation |
| `demo.py` | Runs both algorithms end-to-end with a printed step-by-step walkthrough |
| `test_project.py` | Correctness tests, including the two required edge cases |

## Run it

```bash
python3 demo.py            # walkthrough + generates SVGs and kmeans_simulation.html
python3 test_project.py    # correctness checks (or: python3 -m pytest test_project.py -v)
```

`kmeans_simulation.html` animates the centroids moving through each assignment
and update iteration. In Colab or Jupyter, `demo_kmeans()` displays the
animation inline; in a regular terminal, open the generated HTML file in a
browser.

## How each deliverable is met

**1. Distance Metric** — `core.euclidean_distance(a, b)` computes
`d = sqrt(Σ(xᵢ - yᵢ)²)` directly from the coordinates, with no distance
helper from any library.

**2. Custom Sorting** — `core.quicksort()` is a from-scratch Lomuto-partition
quicksort. `KNNClassifier.predict_one()` uses it (not `sorted()`/`.sort()`)
to rank every training point by distance before taking the top K.

**3. Majority Voting** — the top-K labels are tallied in a plain `dict`.
**Tie handling:** if two+ classes get the same vote count, the tie goes to
whichever tied class has the single nearest neighbor (smallest raw
distance) — a geometric tie-break rather than an arbitrary one. See
`test_knn_tie_break_uses_nearest_neighbor` and the edge-case demo output.

**4. K-Means iterative loop** — `KMeans.fit()`:
- **Centroid Init:** K distinct data points are sampled as starting centers.
- **Assignment Step:** every point is assigned to its nearest centroid by
  Euclidean distance.
- **Update Step:** each centroid becomes the mean of its assigned points.
- Repeats until centroid movement drops below `tol` or `max_iters` is hit.

**Edge case — empty clusters:** if an assignment step leaves a centroid with
zero points, the mean update is undefined (0/0). Instead of crashing,
`_update_centroids()` reseeds that centroid at whichever data point is
currently farthest from its own assigned centroid (the worst-fit point in
the dataset), giving the empty cluster a useful new starting position. See
`test_kmeans_handles_empty_cluster_without_crashing` and the edge-case demo
output.

**5. Code Clarity & Walkthrough** — `demo.py` prints the pipeline for a
sample query point (distances → sorted ranking → vote tally → decision) and
for K-Means (final centroids, cluster sizes), plus a dedicated section
demonstrating both required edge cases with explanations of the rule
applied.

## Design notes worth discussing in review

- `quicksort()` returns a new list rather than sorting in place, so the
  original training data is never mutated by a prediction call.
- KNN is "lazy": `fit()` just stores the data; all computation happens in
  `predict()`. This matches how KNN is actually defined (no real training
  phase).
- K-Means centroid init uses random sampling of existing data points
  (not fully random coordinates) — this keeps starting centroids inside the
  actual data distribution and avoids degenerate empty clusters most of the
  time (though the empty-cluster path is still implemented and tested).
- Convergence is judged by centroid movement (`tol`), not by "assignments
  unchanged," so it also handles the case where assignments flip-flop
  between two centroids that are essentially equidistant.
