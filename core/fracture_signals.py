"""
Five independent fracture-detection signals for experimental comparison.

Each function returns a 1-D numpy array of length = len(windows), values in [0, 1].
Higher value = more fractured (except colour_oxidation: higher = more natural).
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops


# ── 1. Hough line density ─────────────────────────────────────────────────────

def hough_line_signal(img, mask, windows):
    """
    Detects linear features (cracks) using probabilistic Hough transform.
    Returns per-window mean line density, normalised to [0, 1].
    Mineral grains are compact — they don't form lines spanning the core height.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges[mask == 0] = 0

    min_line_len = int(img.shape[0] * 0.30)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=20,
        minLineLength=min_line_len,
        maxLineGap=10,
    )

    # Build per-column line-hit density
    col_density = np.zeros(img.shape[1], dtype=float)
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            col_density[min(x1, x2): max(x1, x2) + 1] += 1.0

    # Aggregate to per-window
    signal = np.array([col_density[x1:x2].mean() for x1, x2 in windows])
    vmax   = signal.max()
    return signal / vmax if vmax > 0 else signal


# ── 2. GLCM texture (inverted homogeneity) ────────────────────────────────────

def glcm_texture_signal(img, mask, windows):
    """
    Computes GLCM homogeneity per window (4 angles, distance=1).
    Result is inverted: high value = low homogeneity = disrupted texture = fractured.
    Intact rock has uniform, repetitive texture; fracture zones disrupt it.
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    signal = np.zeros(len(windows))

    for i, (x1, x2) in enumerate(windows):
        strip = gray[:, x1:x2].copy()
        msk   = mask[:, x1:x2]
        rock  = msk > 0

        if rock.sum() < 50:
            signal[i] = 0.5
            continue

        # Quantise to 8 levels for speed; mask non-rock as 0
        strip_q = np.clip(strip // 32, 0, 7).astype(np.uint8)
        strip_q[~rock] = 0

        glcm = graycomatrix(
            strip_q,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=8,
            symmetric=True,
            normed=True,
        )
        homogeneity = float(graycoprops(glcm, "homogeneity").mean())
        signal[i]   = 1.0 - homogeneity   # invert: high → more fractured

    return signal


# ── 3. Colour oxidation (natural fracture indicator) ──────────────────────────

def colour_oxidation_signal(img_raw, mask, windows):
    """
    Fraction of rock pixels with oxidation colour (orange/brown, HSV hue 5–25).
    Higher → more natural fracture (weathered surface).
    Operates on the original image BEFORE CLAHE to preserve colour fidelity.
    Normalised to [0, 1] relative to the maximum found in this row.
    """
    hsv  = cv2.cvtColor(img_raw, cv2.COLOR_RGB2HSV)
    h_ch = hsv[:, :, 0].astype(float)   # OpenCV hue: 0–180
    s_ch = hsv[:, :, 1].astype(float)

    # Orange/rust: hue 8–20 (strict — avoids pinkish feldspars), high saturation
    oxidised = ((h_ch >= 8) & (h_ch <= 20) & (s_ch >= 55)).astype(float)

    signal = np.zeros(len(windows))
    for i, (x1, x2) in enumerate(windows):
        rock = mask[:, x1:x2] > 0
        if rock.sum() == 0:
            continue
        signal[i] = float(oxidised[:, x1:x2][rock].mean())

    vmax = signal.max()
    return signal / vmax if vmax > 0 else signal


# ── 4. Dark-region eccentricity (shape of dark blobs) ────────────────────────

def morphology_crack_signal(img, mask, windows):
    """
    Mean eccentricity of dark connected regions within each window.
    Eccentricity near 1 = elongated/linear → crack-like.
    Eccentricity near 0 = compact/round   → mineral grain-like.
    Threshold: rock-median × 0.65 defines 'dark'.
    """
    gray      = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rock_gray = gray.astype(float)
    rock_gray[mask == 0] = np.nan

    img_median = float(np.nanmedian(rock_gray))
    if img_median < 15:
        return np.full(len(windows), 0.5)

    dark_thresh = img_median * 0.65
    dark_bin    = ((gray < dark_thresh) & (mask > 0)).astype(np.uint8)

    labeled = label(dark_bin, connectivity=2)
    props   = regionprops(labeled)

    if not props:
        return np.full(len(windows), 0.5)

    region_cx    = np.array([p.centroid[1] for p in props])
    region_eccen = np.array([p.eccentricity for p in props])
    region_area  = np.array([p.area for p in props])

    MIN_AREA = 20
    signal   = np.zeros(len(windows))

    for i, (x1, x2) in enumerate(windows):
        in_win = (region_cx >= x1) & (region_cx < x2) & (region_area >= MIN_AREA)
        signal[i] = float(region_eccen[in_win].mean()) if in_win.any() else 0.5

    return signal


# ── 5. Sobel gradient — hairline crack detector ───────────────────────────────

# ── Natural vs. mechanical fracture origin ────────────────────────────────────

def fracture_origin(img_raw, mask, zone):
    """
    Estimate natural vs. mechanical origin for a single fractured/rubble zone.

    Signals used
    ------------
    1. Oxidation colour — HSV hue 8–20 (strict orange), S≥55 → natural.
       Tighter than before: avoids flagging pinkish/beige granite feldspars
       (hue 5–8, low saturation) as "oxidised".
    2. Surface freshness — high V channel brightness → mechanical.
       Fresh drill-induced breaks expose unweathered, bright rock faces.

    Formula logic
    -------------
    • Oxidation is the ONLY direct chemical evidence of weathering → primary signal.
    • Zero oxidation → mechanical regardless of brightness.
    • High freshness + zero oxidation → strongly mechanical.
    • Dark + zero oxidation → mechanical (dark ≠ natural; drill cores are dark too).
    • High oxidation overrides freshness toward natural.

    Returns
    -------
    p_natural    : float  0–1
    p_mechanical : float  0–1  (= 1 − p_natural)
    confidence   : str    'alta' | 'media' | 'baja'
    details      : dict   {'oxid_pct', 'mean_v', 'fresh_norm'}
    """
    import cv2 as _cv2
    hsv  = _cv2.cvtColor(img_raw, _cv2.COLOR_RGB2HSV).astype(float)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    x1, x2 = zone["x_start"], zone["x_end"]
    rock    = mask[:, x1:x2] > 0

    if rock.sum() == 0:
        return 0.5, 0.5, "baja", {"oxid_pct": 0.0, "mean_v": 0.0, "fresh_norm": 0.0}

    # 1. Oxidation: strict orange/rust (hue 8–20 on OpenCV 0–180 scale, S≥55)
    #    Excludes pinkish granite minerals (hue 2–8, low S) and yellowed tray edges.
    oxidised  = (
        (h_ch[:, x1:x2] >= 8) & (h_ch[:, x1:x2] <= 20) & (s_ch[:, x1:x2] >= 55)
    )
    oxid_frac = float(oxidised[rock].mean())           # 0–1
    oxid_norm = min(1.0, oxid_frac / 0.15)            # 15 % stained → full natural score

    # 2. Surface freshness: meaningfully bright above mean_v=100 (0→1 at 200)
    mean_v     = float(v_ch[:, x1:x2][rock].mean())   # 0–255
    fresh_norm = float(max(0.0, min(1.0, (mean_v - 100) / 100)))

    # Combine ─────────────────────────────────────────────────────────────────
    # p_natural is DRIVEN by oxidation. Freshness can only reduce it when
    # oxidation is absent (fresh surface with no staining = mechanical).
    p_natural = float(min(1.0, max(0.0,
        0.90 * oxid_norm
        - 0.30 * fresh_norm * (1.0 - oxid_norm)
    )))
    p_mech = 1.0 - p_natural

    # Confidence: strong when at least one signal is clear
    if oxid_frac > 0.10 or fresh_norm > 0.55:
        confidence = "alta"
    elif oxid_frac > 0.03 or fresh_norm > 0.25:
        confidence = "media"
    else:
        confidence = "baja"   # both signals weak — zone is ambiguous

    details = {
        "oxid_pct":  round(oxid_frac * 100, 1),
        "mean_v":    round(mean_v, 1),
        "fresh_norm": round(fresh_norm, 2),
    }
    return round(p_natural, 3), round(p_mech, 3), confidence, details


def sobel_gradient_signal(img, mask, windows):
    """
    Per-window Sobel magnitude outlier fraction, normalised to [0, 1].

    Uses gradient magnitude (√Sx²+Sy²) so cracks at any angle are detected.
    Flags columns whose max gradient is a statistical outlier within the window
    — works for diagonal cracks that only cross 1–2 pixels per column.
    """
    gray     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mg = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mg[mask == 0] = 0.0

    signal = np.zeros(len(windows))
    for i, (x1, x2) in enumerate(windows):
        col_max = sobel_mg[:, x1:x2].max(axis=0).astype(float)
        valid   = col_max[col_max > 0]
        if len(valid) > 3:
            cm_mean = valid.mean()
            cm_std  = valid.std()
            if cm_std > 1e-3:
                signal[i] = float((col_max > cm_mean + 1.5 * cm_std).mean())

    vmax = signal.max()
    return signal / vmax if vmax > 0 else signal


def sobel_gradient_image(img, mask):
    """
    Return Sobel magnitude as a normalised uint8 image (for display).
    Non-rock pixels are set to 0.
    """
    gray     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mg = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mg[mask == 0] = 0.0
    vmax = sobel_mg.max()
    if vmax > 0:
        sobel_mg = (sobel_mg / vmax * 255).astype(np.uint8)
    return sobel_mg.astype(np.uint8)
