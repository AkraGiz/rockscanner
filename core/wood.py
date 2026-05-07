import cv2
import numpy as np


def detect_wood_blocks(img, mask,
                       brightness_min=200,
                       uniformity_max=80,    # contrast_min
                       min_width_ratio=0.005,
                       min_height_ratio=0.30,
                       bimodal_min=0.35):
    """
    Detect wooden separator blocks directly from the image — mask-independent.

    Wood blocks with handwritten text have a very distinctive per-column
    brightness distribution:
      - Many very BRIGHT pixels  → paper / wood background  (high p90)
      - Some very DARK pixels    → ink / text               (low p10)
      - HIGH contrast = p90 − p10

    Any rock type (grey, teal, beige, red …) has a much more compact
    distribution: p90 and p10 are relatively close together.

    Parameters
    ----------
    brightness_min : int
        Minimum p90 value for a column to be considered a wood candidate.
        (the paper background must be this bright)
    uniformity_max : int
        Minimum p90−p10 contrast required.
        Higher → only very high-contrast columns (text on paper) qualify.
    min_width_ratio : float
        Minimum block width as fraction of image width.
    min_height_ratio : float
        Minimum fraction of column pixels that must be "bright" (paper).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    contrast_min = uniformity_max   # rename for clarity inside this function

    # ── Masked grayscale: exclude tray / background pixels ───────────────────
    # The rock mask correctly marks the wooden tray as background (mask==0).
    # Computing percentiles only on mask==255 pixels removes the tray signal.
    # The wood block itself is typically mis-classified as "rock" by the mask,
    # so it IS included — which is exactly what we want.
    masked_gray = gray.astype(float)
    masked_gray[mask == 0] = np.nan

    # ── Per-column percentiles (rock + wood pixels only) ─────────────────────
    col_p90 = np.nanpercentile(masked_gray, 90, axis=0)
    col_p10 = np.nanpercentile(masked_gray, 10, axis=0)
    # Columns with no valid pixels → treat as "not wood"
    col_p90 = np.where(np.isnan(col_p90), 0.0, col_p90)
    col_p10 = np.where(np.isnan(col_p10), 0.0, col_p10)
    col_contrast = col_p90 - col_p10

    # ── Wood-candidate columns ────────────────────────────────────────────────
    wood_col = (col_p90 >= brightness_min) & (col_contrast >= contrast_min)

    # Exclude leftmost / rightmost border columns
    margin = max(5, int(w * 0.01))
    wood_col[:margin]  = False
    wood_col[-margin:] = False

    # Safety check: if more than 40 % of columns qualify → false positive storm
    if wood_col.mean() > 0.40:
        return []

    # ── Group consecutive candidate columns ───────────────────────────────────
    min_w_px   = max(5, int(w * min_width_ratio))   # e.g. ≥22px on 4300px image
    max_w_px   = int(w * 0.15)          # a real block is never > 15 % of row width
    candidates = _group_consecutive(wood_col, min_width=min_w_px)

    # Global p90 median: the wood block must be brighter than the typical column
    global_p90_median = float(np.median(col_p90[col_p90 > 0])) if (col_p90 > 0).any() else 0.0

    # ── Validate each candidate ───────────────────────────────────────────────
    flank = max(10, int(w * 0.02))      # how far to look left/right for rock context

    result = []
    for x1, x2 in candidates:
        if (x2 - x1) > max_w_px:
            continue                    # too wide → likely tray remnant

        strip_raw = gray[:, x1:x2]

        # Vertical coverage: bright pixels must appear in enough rows
        rows_with_bright = (strip_raw >= brightness_min * 0.75).any(axis=1)
        if float(rows_with_bright.mean()) < min_height_ratio:
            continue

        # Context check: the candidate must be a clear LOCAL PEAK.
        # Compare mean p90 of the block vs mean p90 of each flank window.
        # A real wood block spikes sharply above surrounding rock.
        # Flat elevated zones (bright rock) have low contrast with neighbors.
        block_p90_mean = float(col_p90[x1:x2].mean())

        left_slice  = col_p90[max(0, x1 - flank): x1]
        right_slice = col_p90[x2: min(w, x2 + flank)]

        left_mean  = float(left_slice.mean())  if len(left_slice)  > 0 else block_p90_mean
        right_mean = float(right_slice.mean()) if len(right_slice) > 0 else block_p90_mean

        # How much brighter is the block than each neighbor?
        left_peak_contrast  = block_p90_mean - left_mean
        right_peak_contrast = block_p90_mean - right_mean

        # Must be brighter than the typical column in this row
        if block_p90_mean < global_p90_median + 10.0:
            continue

        # At least one side must show a clear brightness jump (≥ 20 units)
        min_peak_contrast = 20.0
        if not (left_peak_contrast >= min_peak_contrast or
                right_peak_contrast >= min_peak_contrast):
            continue

        result.append({"x_start": int(x1), "x_end": int(x2)})

    return _merge_adjacent(result, gap=15)


# ── Proximity filter: require mutual corroboration ───────────────────────────

def filter_wood_by_proximity(wood_blocks, w, row_length_cm, max_dist_cm=50.0):
    """
    Consolidate wood detections by proximity.

    Rules (since core runs are 3 m → at most 1 real block per 1-m row):
      - 0 detections  → nothing.
      - 1 detection   → keep it as-is (lone blocks are valid).
      - 2+ detections within max_dist_cm of each other → keep only the
        widest one (most signal coverage = most confident).
      - 2+ detections all separated by more than max_dist_cm → keep each
        as-is (they are in distinct positions, treated independently).

    Result: 0 or 1 wood block per cluster of nearby detections.
    """
    if not wood_blocks:
        return []
    if len(wood_blocks) == 1:
        return wood_blocks

    px_per_cm   = w / row_length_cm
    max_dist_px = max_dist_cm * px_per_cm

    # Single-linkage clustering by centre distance
    blocks  = sorted(wood_blocks, key=lambda b: b["x_start"])
    visited = [False] * len(blocks)
    clusters = []

    for i, wb in enumerate(blocks):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        cx_i = (wb["x_start"] + wb["x_end"]) / 2.0
        for j, wb2 in enumerate(blocks):
            if visited[j]:
                continue
            cx_j = (wb2["x_start"] + wb2["x_end"]) / 2.0
            if abs(cx_i - cx_j) <= max_dist_px:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)

    # From each cluster keep the widest block (= most confident detection)
    result = []
    for cluster in clusters:
        members = [blocks[i] for i in cluster]
        best = max(members, key=lambda b: b["x_end"] - b["x_start"])
        result.append(best)

    return result


# ── Diagnostic: column profiles ──────────────────────────────────────────────

def wood_column_profiles(img, mask):
    """
    Return per-column arrays useful for debugging wood block detection.
    All arrays have length = image width.
    """
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    masked = gray.astype(float)
    masked[mask == 0] = np.nan

    col_p90      = np.nanpercentile(masked, 90, axis=0)
    col_p10      = np.nanpercentile(masked, 10, axis=0)
    col_p90      = np.where(np.isnan(col_p90), 0.0, col_p90)
    col_p10      = np.where(np.isnan(col_p10), 0.0, col_p10)
    col_contrast = col_p90 - col_p10

    # Bimodality per column
    bimodal = np.zeros(img.shape[1])
    for x in range(img.shape[1]):
        col = gray[:, x].astype(float)
        valid = mask[:, x] > 0
        if valid.any():
            pix = col[valid]
            bimodal[x] = ((pix > 215) | (pix < 55)).mean()

    return {
        "col_p90":      col_p90,
        "col_p10":      col_p10,
        "col_contrast": col_contrast,
        "bimodal":      bimodal,
    }


# ── Public helper: insert wood zones into the zone list ───────────────────────

def apply_wood_to_zones(zones, wood_blocks, img_width, row_length_cm=None):
    """
    Insert wood block zones into the zone list, trimming rock zones where
    a wood block sits.  Returns a new, position-sorted zone list.
    """
    if not wood_blocks:
        return zones

    result = []
    for zone in zones:
        segments = [(zone["x_start"], zone["x_end"])]
        for wb in wood_blocks:
            new_segs = []
            for s, e in segments:
                if wb["x_end"] <= s or wb["x_start"] >= e:
                    new_segs.append((s, e))
                else:
                    if s < wb["x_start"]:
                        new_segs.append((s, wb["x_start"]))
                    if wb["x_end"] < e:
                        new_segs.append((wb["x_end"], e))
            segments = new_segs
        for s, e in segments:
            if e > s:
                result.append(_zone_dict(zone["label"], s, e, img_width, row_length_cm))

    for wb in wood_blocks:
        result.append(_zone_dict("wood", wb["x_start"], wb["x_end"], img_width, row_length_cm))

    result.sort(key=lambda z: z["x_start"])
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _group_consecutive(bool_arr, min_width=5):
    """Return list of (start, end) for consecutive True runs ≥ min_width."""
    groups, in_group, start = [], False, 0
    for i, val in enumerate(bool_arr):
        if val and not in_group:
            start, in_group = i, True
        elif not val and in_group:
            if i - start >= min_width:
                groups.append((start, i))
            in_group = False
    if in_group and len(bool_arr) - start >= min_width:
        groups.append((start, len(bool_arr)))
    return groups


def _merge_adjacent(blocks, gap=15):
    if not blocks:
        return []
    blocks = sorted(blocks, key=lambda b: b["x_start"])
    merged = [dict(blocks[0])]
    for b in blocks[1:]:
        if b["x_start"] <= merged[-1]["x_end"] + gap:
            merged[-1]["x_end"] = max(merged[-1]["x_end"], b["x_end"])
        else:
            merged.append(dict(b))
    return merged


def _zone_dict(label, x_start, x_end, img_width, row_length_cm):
    pct_s = x_start / img_width
    pct_e = x_end   / img_width
    return {
        "label":     label,
        "x_start":   x_start,
        "x_end":     x_end,
        "pct_start": pct_s,
        "pct_end":   pct_e,
        "cm_start":  pct_s * row_length_cm if row_length_cm else None,
        "cm_end":    pct_e * row_length_cm if row_length_cm else None,
    }
