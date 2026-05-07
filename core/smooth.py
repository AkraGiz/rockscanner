import numpy as np

# Severity order: intact < fractured < rubble
# Smoothing may only move a window toward equal or higher severity, never softer.
LABEL_TO_INT = {"intact": 0, "fractured": 1, "rubble": 2}
INT_TO_LABEL = {0: "intact",  1: "fractured",  2: "rubble"}


def smooth_labels(labels, window_size=3, min_zone_windows=2):
    """
    Reduce noise without destroying narrow fracture zones.

    Steps:
      1. Remove truly isolated single-window outliers only
         (a lone window whose both neighbors agree on a different label).
      2. Absorb zones smaller than min_zone_windows into their best neighbor.

    Deliberately conservative: a 2-window fractured zone between intact blocks
    is preserved, because real fractures CAN be narrow.
    """
    if not labels:
        return labels

    arr = np.array([LABEL_TO_INT[l] for l in labels])

    # Step 1 – kill isolated single-window noise only
    arr = _remove_isolated_singles(arr)

    # Step 2 – absorb zones below minimum size
    arr = _absorb_small_zones(arr, min_zone_windows)

    return [INT_TO_LABEL[int(v)] for v in arr]


def _remove_isolated_singles(arr):
    """
    Flip a window only when it is completely alone AND the flip moves it toward
    a more severe (or equal) classification — never toward a softer one.

    Rule: intact(0) < fractured(1) < rubble(2).
    A lone rubble window is never flipped to fractured/intact.
    A lone fractured window is never flipped to intact.
    Only upward noise (e.g. a lone intact inside rubble) is corrected.
    """
    result = arr.copy()
    for i in range(1, len(arr) - 1):
        if arr[i - 1] == arr[i + 1] and arr[i] != arr[i - 1]:
            target = arr[i - 1]
            if target >= arr[i]:   # only allow flipping toward equal or higher severity
                result[i] = target
    return result


def _absorb_small_zones(arr, min_size):
    """Repeatedly merge zones shorter than min_size into a neighbor."""
    result = arr.copy()
    changed = True
    while changed:
        changed = False
        zones = _get_zones(result)
        for i, (label, start, end) in enumerate(zones):
            if (end - start) < min_size:
                if len(zones) == 1:
                    break  # only one zone, nothing to merge into
                if i == 0:
                    target = zones[1][0]
                elif i == len(zones) - 1:
                    target = zones[-2][0]
                else:
                    prev_len = zones[i - 1][2] - zones[i - 1][1]
                    next_len = zones[i + 1][2] - zones[i + 1][1]
                    target = zones[i - 1][0] if prev_len >= next_len else zones[i + 1][0]
                # Only absorb if target is equal or more severe — never soften
                if target < label:
                    continue
                result[start:end] = target
                changed = True
                break   # restart after every merge
    return result


def get_zones(labels):
    """Public wrapper: return list of (label_str, start_idx, end_idx)."""
    arr = np.array([LABEL_TO_INT[l] for l in labels])
    return [(INT_TO_LABEL[lbl], s, e) for lbl, s, e in _get_zones(arr)]


def _get_zones(arr):
    """Return list of (int_label, start, end) for contiguous runs."""
    if len(arr) == 0:
        return []
    zones = []
    cur, start = arr[0], 0
    for i in range(1, len(arr)):
        if arr[i] != cur:
            zones.append((int(cur), start, i))
            cur, start = arr[i], i
    zones.append((int(cur), start, len(arr)))
    return zones
