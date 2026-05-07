import cv2
import numpy as np


def compute_window_features(img_strip, mask_strip, sobel_sigma=2.5, sobel_weight=0.25):
    """
    Compute continuity/fragmentation metrics for a single vertical strip.

    Features fall into two groups:
      A) Mass-based  — work well when rock is fragmented into separate pieces
      B) Crack-based — work well for tightly-cropped images where the rock
                       fills the frame and fractures appear as dark columns
    """
    h, w = mask_strip.shape
    total_px = h * w
    rock_px  = (mask_strip > 0).astype(np.uint8)
    rock_count = int(rock_px.sum())

    # Compute grayscale once, reused by multiple features
    gray = cv2.cvtColor(img_strip, cv2.COLOR_RGB2GRAY)

    feats = {}

    # ── A1. Rock occupancy ───────────────────────────────────────────────────
    feats["rock_occupancy"] = rock_count / total_px if total_px > 0 else 0.0

    if rock_count == 0:
        feats.update({
            "connected_components_count": 0,
            "largest_component_ratio":    0.0,
            "width_stability":            0.0,
            "internal_edge_density":      0.0,
            "contour_roughness":          1.0,
            "crack_column_fraction":      1.0,
            "brightness_cv":              1.0,
            "local_fragmentation":        1.0,
            "longitudinal_continuity":    0.0,
        })
        return feats

    # ── A2. Connected components ─────────────────────────────────────────────
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(rock_px, connectivity=8)
    num_comp = num_labels - 1
    feats["connected_components_count"] = num_comp

    # ── A3. Largest component ratio ──────────────────────────────────────────
    if num_comp > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        feats["largest_component_ratio"] = float(areas.max()) / rock_count
    else:
        feats["largest_component_ratio"] = 0.0

    # ── A4. Width stability ──────────────────────────────────────────────────
    row_coverage = rock_px.sum(axis=1).astype(float)
    mean_cov = row_coverage.mean()
    feats["width_stability"] = float(
        1.0 - min(1.0, row_coverage.std() / mean_cov)
    ) if mean_cov > 0 else 0.0

    # ── A5. Internal edge density ────────────────────────────────────────────
    edges = cv2.Canny(gray, 30, 90)
    internal_edges = int(((edges > 0) & (rock_px > 0)).sum())
    feats["internal_edge_density"] = min(1.0, internal_edges / rock_count)

    # ── A6. Contour roughness ────────────────────────────────────────────────
    contours, _ = cv2.findContours(rock_px, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        total_perim = sum(cv2.arcLength(c, True) for c in contours)
        smooth_est  = 4.0 * np.sqrt(max(rock_count, 1))
        ratio = total_perim / smooth_est
        feats["contour_roughness"] = float(min(1.0, (ratio - 1.0) / 5.0))
    else:
        feats["contour_roughness"] = 0.0

    # ── B1. Crack detection — strict-percentile approach ─────────────────────
    #
    # A real fracture (shadow/gap) pulls the low-percentile of its column very
    # close to 0. A dark mineral grain also lowers the percentile, but not
    # nearly as far — the rest of the column still has bright rock pixels.
    #
    # Fix vs. original: threshold multiplier tightened from 0.75 → 0.52.
    # A column must be dramatically darker than the median column to be flagged
    # as a crack. Mineral-grain columns are typically at 60–90 % of median,
    # well above the new threshold; fracture-shadow columns sit at 5–25 %.
    #
    # The p15 percentile is kept (rather than median) so diagonal cracks — which
    # darken only a fraction of a column's height — are still detected: even a
    # thin diagonal line pushes the 15th-percentile sharply downward.

    CRACK_THRESH_RATIO = 0.52   # fraction of median; lower → stricter

    masked_gray = gray.astype(float)
    masked_gray[mask_strip == 0] = np.nan

    col_dark = np.nanpercentile(masked_gray, 15, axis=0)
    col_dark = np.where(np.isnan(col_dark), 255.0, col_dark)

    row_dark = np.nanpercentile(masked_gray, 15, axis=1)
    row_dark = np.where(np.isnan(row_dark), 255.0, row_dark)

    median_col = float(np.median(col_dark))
    median_row = float(np.median(row_dark))

    if median_col > 15:
        thresh_col     = median_col * CRACK_THRESH_RATIO
        crack_col_frac = float((col_dark < thresh_col).mean())
        bri_cv         = float(min(1.0, col_dark.std() / (median_col + 1e-6)))
    else:
        crack_col_frac = 0.0
        bri_cv         = 0.0

    if median_row > 15:
        thresh_row     = median_row * CRACK_THRESH_RATIO
        crack_row_frac = float((row_dark < thresh_row).mean())
    else:
        crack_row_frac = 0.0

    # ── B2. Sobel magnitude — diagonal & hairline crack detector ─────────────
    # Uses gradient MAGNITUDE (√Sx²+Sy²) so diagonal cracks at any angle are
    # captured equally — no orientation bias.
    #
    # A diagonal crack only crosses 1–2 pixels per column, so "coverage" checks
    # fail. Instead: flag columns whose MAXIMUM gradient is a statistical outlier
    # vs. the rest of the strip. Rock texture produces uniform column maxima;
    # a crack creates one very bright outlier column.

    sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mg = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mg[mask_strip == 0] = 0.0

    col_max = sobel_mg.max(axis=0).astype(float)   # strongest edge per column

    valid_cols = col_max[col_max > 0]
    if len(valid_cols) > 5:
        cm_mean = float(valid_cols.mean())
        cm_std  = float(valid_cols.std())
        if cm_std > 1e-3:
            sobel_crack_frac = float((col_max > cm_mean + sobel_sigma * cm_std).mean())
        else:
            sobel_crack_frac = 0.0
    else:
        sobel_crack_frac = 0.0

    # Weighted combination: brightness signal + Sobel signal
    brightness_crack = max(crack_col_frac, crack_row_frac)
    combined_crack   = (1.0 - sobel_weight) * brightness_crack + sobel_weight * sobel_crack_frac
    feats["crack_column_fraction"] = float(min(1.0, combined_crack))
    feats["brightness_cv"]         = bri_cv

    # ── Aggregate fragmentation score ────────────────────────────────────────
    # Blend mass-based and crack-based signals equally so the pipeline works
    # for both tightly-cropped rows (crack-based dominates) and
    # fragmented rubble (mass-based dominates).
    n_norm   = min(1.0, num_comp / 8.0)
    lr_inv   = 1.0 - feats["largest_component_ratio"]
    occ_inv  = 1.0 - feats["rock_occupancy"]
    edge_d   = feats["internal_edge_density"]
    rough    = feats["contour_roughness"]
    crack_f  = feats["crack_column_fraction"]
    bri_cv   = feats["brightness_cv"]

    feats["local_fragmentation"] = float(min(1.0,
        0.10 * n_norm  +
        0.10 * lr_inv  +
        0.08 * occ_inv +
        0.12 * edge_d  +
        0.08 * rough   +
        0.35 * crack_f +   # crack-based — primary signal for solid-rock rows
        0.17 * bri_cv
    ))

    # longitudinal_continuity filled in by compute_all_window_features
    feats["longitudinal_continuity"] = 0.5

    return feats


def compute_all_window_features(img, mask, windows, sobel_sigma=2.5, sobel_weight=0.25):
    """
    Compute features for every window, then fill longitudinal_continuity.
    Returns list of feature dicts (same order as windows).
    """
    all_feats = []
    profiles  = []

    for x1, x2 in windows:
        feats = compute_window_features(
            img[:, x1:x2], mask[:, x1:x2],
            sobel_sigma=sobel_sigma,
            sobel_weight=sobel_weight,
        )
        all_feats.append(feats)
        profiles.append((mask[:, x1:x2] > 0).sum(axis=1).astype(float))

    n = len(all_feats)
    for i in range(n):
        scores = []
        for j in [i - 1, i + 1]:
            if not (0 <= j < n):
                continue
            a, b = profiles[i], profiles[j]
            if a.std() > 0 and b.std() > 0:
                scores.append(max(0.0, float(np.corrcoef(a, b)[0, 1])))
            else:
                scores.append(1.0 if (a.mean() > 0 and b.mean() > 0) else 0.0)
        all_feats[i]["longitudinal_continuity"] = float(np.mean(scores)) if scores else 0.5

    return all_feats
