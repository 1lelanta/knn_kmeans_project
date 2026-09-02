
import math


def euclidean_distance(a, b):
    """Straight-line distance between two points of equal dimensionality.
    """
    if len(a) != len(b):
        raise ValueError("Points must have the same number of dimensions")
    total = 0.0
    for ai, bi in zip(a, b):
        total += (ai - bi) ** 2
    return math.sqrt(total)


def quicksort(items, key=lambda x: x):
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
