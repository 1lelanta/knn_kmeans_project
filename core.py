"""
core.py
--------
Zero-dependency building blocks for both algorithms:
  - euclidean_distance:  d = sqrt(sum((xi - yi)^2))
  - quicksort:           a from-scratch sort (built-in sort()/sorted() are banned
                          for the KNN ranking step)

Only the standard `math` module is used, purely for sqrt() — that is basic
arithmetic, not a machine-learning helper, so it does not violate the
"no external libraries" rule (no numpy / pandas / sklearn anywhere).
"""

import math


def euclidean_distance(a, b):
    """Straight-line distance between two points of equal dimensionality.

    d = sqrt( sum_i (a_i - b_i)^2 )
    """
    if len(a) != len(b):
        raise ValueError("Points must have the same number of dimensions")
    total = 0.0
    for ai, bi in zip(a, b):
        total += (ai - bi) ** 2
    return math.sqrt(total)


def quicksort(items, key=lambda x: x):
    """Ascending quicksort implemented from scratch (Lomuto partition).

    `items` is not mutated; a new sorted list is returned. `key` extracts the
    comparison value from each element (mirrors the built-in sort's `key=`,
    without relying on the built-in sort itself).
    """
    if len(items) <= 1:
        return list(items)

    items = list(items)
    _quicksort_inplace(items, 0, len(items) - 1, key)
    return items


def _quicksort_inplace(arr, low, high, key):
    if low >= high:
        return
    pivot_index = _partition(arr, low, high, key)
    _quicksort_inplace(arr, low, pivot_index - 1, key)
    _quicksort_inplace(arr, pivot_index + 1, high, key)


def _partition(arr, low, high, key):
    pivot = key(arr[high])
    i = low - 1
    for j in range(low, high):
        if key(arr[j]) <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
