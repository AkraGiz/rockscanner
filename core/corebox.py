"""
Photo Core detection — automatic box outline + horizontal divider detection.

Phase 1 scope: standard wood box with horizontal channels, dry rocks.
All other cases (vertical, plastic, metal tubes, wet) are routed through
the manual editor instead.

Public API
----------
detect_box_outline(img)            → 4-vertex polygon (np.ndarray Nx2)
detect_horizontal_dividers(img, b) → list of Y-coordinates of divider strips
extract_rows(img, box, dividers)   → list[Row] in reading order

All coordinates are in the original image pixel space.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    """A single horizontal channel extracted from a Photo Core."""
    img: np.ndarray                       # RGB strip image
    bbox: tuple[int, int, int, int]       # (x1, y1, x2, y2) in original photo
    index: int                            # 0-based reading order


# ─────────────────────────────────────────────────────────────────────────────
# 0 · Auto-deskew (rotate the image so horizontal features are horizontal)
# ─────────────────────────────────────────────────────────────────────────────

def detect_skew_angle(img: np.ndarray, max_angle_deg: float = 20.0) -> float:
    """
    Estimate the rotation of horizontal features (box edges, listones).

    Returns an angle in degrees. Positive = features tilt down-right from
    horizontal (image is rotated CW); negative = up-right (image CCW).

    Uses probabilistic Hough on Canny edges, restricted to lines roughly
    horizontal (|θ| < max_angle_deg).  Median of detected angles is robust
    against noise.  Image is downscaled to ≤1500 px for speed.
    """
    h, w = img.shape[:2]
    if h < 50 or w < 50:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    # Downscale for speed — accuracy doesn't suffer for skew detection
    long_side = max(h, w)
    if long_side > 1500:
        s = 1500 / long_side
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(gray, 60, 180)
    min_len = max(50, gray.shape[1] // 4)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360,
        threshold=120, minLineLength=min_len, maxLineGap=25,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == x1:
            continue
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Normalize to [-90, 90]
        if a > 90:
            a -= 180
        elif a < -90:
            a += 180
        if abs(a) <= max_angle_deg:
            angles.append(a)

    if not angles:
        return 0.0
    # Robust central tendency
    return float(np.median(angles))


def deskew_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate image so that detected horizontal features become truly horizontal.

    Pass the angle returned by `detect_skew_angle` — rotation direction is
    handled internally.  Output is enlarged to fit the rotated content
    without cropping; the new background is filled with white.
    """
    if abs(angle_deg) < 0.3:
        return img   # negligible — skip the resampling cost

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # Rotate in the SAME direction as the detected angle to undo the skew.
    # (OpenCV's image Y axis points down so its "positive angle = CCW" maps
    # to visual CW; rotating by +angle compensates the skew detected as
    # +angle in the same convention.)
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Box outline detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_box_outline(img: np.ndarray) -> np.ndarray:
    """
    Find the outer rectangle of the core box.

    Heuristic
    ---------
    1. Convert to grayscale and blur.
    2. Adaptive threshold to separate "box+contents" (textured) from
       "background" (smooth — table, floor, sky).
    3. Largest connected component → its bounding rectangle.
    4. Try polygon approximation (4 vertices). Fallback to bounding rect.

    Returns
    -------
    np.ndarray of shape (4, 2) with the four corners of the box,
    ordered TL, TR, BR, BL (clockwise from top-left).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Otsu on the blurred image: separates the textured box from smooth bg
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological closing to consolidate the box region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Largest contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: full image
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.05 * h * w:
        # Largest blob is too small to be a real box — fallback to full image
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)

    # Try polygon approximation for a tight rectangle
    perim = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perim, True)
    if len(approx) == 4:
        pts = approx.reshape(-1, 2)
        return _order_corners(pts)

    # Fallback: axis-aligned bounding rect
    x, y, bw, bh = cv2.boundingRect(largest)
    return np.array(
        [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
        dtype=np.int32,
    )


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 points sorted TL, TR, BR, BL (clockwise from top-left)."""
    pts = pts.astype(np.float32)
    # TL has smallest x+y, BR largest x+y. TR has smallest y-x, BL largest y-x.
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Horizontal divider detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_horizontal_dividers(
    img: np.ndarray,
    box: np.ndarray,
    *,
    expected_min_channels: int = 3,
    expected_max_channels: int = 15,
    sensitivity: float = 0.5,
    use_variance: bool = True,
    use_brightness: bool = True,
    use_sobel: bool = True,
    use_continuity: bool = True,
    use_hough: bool = False,
    use_periodicity: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Find Y-coordinates + half-thickness of the strips that separate channels.

    Detection combines several toggleable 1-D signals (each normalised to
    [0,1] and weighted) into a single score, then finds peaks:

      • `use_variance`     — low horizontal variance = uniform divider strip
      • `use_brightness`   — sharp |d(mean brightness)/dy| = content boundary
      • `use_sobel`        — strong horizontal-edge magnitude per row
      • `use_continuity`   — fraction of the width covered by a strong edge
                             (a real listón edge spans the WHOLE box width;
                             a shadow between two rocks is local) ← most precise
      • `use_hough`        — votes from long near-horizontal Hough segments
      • `use_periodicity`  — soft boost assuming rows are roughly equispaced

    All signals are material-agnostic (work for wood, metal, plastic).

    Returns
    -------
    (dividers, thicknesses) — parallel lists. `dividers` are Y centres in
    original-image space; `thicknesses` are measured half-thickness in px.
    """
    x1, y1, x2, y2 = _box_to_bbox(box)
    box_h = y2 - y1
    box_w = x2 - x1
    if box_h < 50 or box_w < 50:
        return [], []

    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop

    # ── Sensitivity-driven parameters ──────────────────────────────────────
    s = float(np.clip(sensitivity, 0.0, 1.0))
    sigma = max(2.0, box_h / (250 - 200 * s))   # ≈ box_h/250 .. box_h/50
    prom_thresh = 0.03 + 0.15 * s               # 0.03 .. 0.18

    def _norm(x):
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    # ── Build the enabled signals (weight, array) ──────────────────────────
    signals: list[tuple[float, np.ndarray]] = []

    # Variance is computed regardless (used for centring later), but only fed
    # into the score if enabled.
    row_std = gray.std(axis=1).astype(float)
    row_std_smooth = gaussian_filter1d(row_std, sigma=sigma)
    if use_variance:
        variance_inv = row_std_smooth.max() - row_std_smooth
        signals.append((0.35, _norm(variance_inv)))

    if use_brightness:
        row_mean = gray.mean(axis=1).astype(float)
        row_mean_smooth = gaussian_filter1d(row_mean, sigma=sigma)
        deriv = np.abs(np.gradient(row_mean_smooth))
        signals.append((0.22, _norm(deriv)))

    # Sobel (shared by both the magnitude signal and the continuity signal)
    sobel_y = None
    if use_sobel or use_continuity:
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    if use_sobel:
        edge_per_row = np.abs(sobel_y).mean(axis=1)
        signals.append((0.18, _norm(gaussian_filter1d(edge_per_row, sigma=sigma))))

    if use_continuity:
        # Fraction of the row width covered by a STRONG horizontal edge.
        # Real listones run edge-to-edge → high fraction.  Inter-rock shadows
        # are local → low fraction even if individually very strong.
        mag = np.abs(sobel_y)
        thr = np.percentile(mag, 85)            # adaptive "strong" threshold
        strong = (mag > max(thr, 1.0)).astype(float)
        continuity = strong.mean(axis=1)        # 0..1 per row
        signals.append((0.30, _norm(gaussian_filter1d(continuity, sigma=sigma))))

    if use_hough:
        signals.append((0.30, _norm(_hough_row_votes(gray, box_w, sigma))))

    if not signals:
        # Nothing enabled → fall back to variance so we always return something
        variance_inv = row_std_smooth.max() - row_std_smooth
        signals.append((1.0, _norm(variance_inv)))

    total_w = sum(w for w, _ in signals)
    score = np.zeros(box_h, dtype=float)
    for w, sig in signals:
        score += (w / total_w) * sig

    # ── Optional periodicity boost (assume roughly equispaced rows) ─────────
    if use_periodicity:
        score = _apply_periodicity_boost(score, min_dist=max(10, box_h // 30))

    # ── Peak detection on the combined score ────────────────────────────────
    min_dist = max(10, box_h // (expected_max_channels + 2))
    peaks, props = find_peaks(score, distance=min_dist, prominence=prom_thresh)

    if len(peaks) < expected_min_channels - 1:
        peaks, props = find_peaks(score, distance=min_dist,
                                  prominence=prom_thresh * 0.5)

    # ── Re-center each peak on the listón band + estimate band thickness ────
    # The raw peak position can sit at the dominant edge of the listón rather
    # than at its geometric centre.  We refine by finding the contiguous band
    # of low row-std around each peak.  The CENTER goes into `dividers`, the
    # measured HALF-THICKNESS (in px) goes into a parallel list.
    rs = row_std_smooth
    median_std = float(np.median(rs))
    centered: list[int] = []
    half_thicknesses: list[int] = []
    for p in peaks:
        peak_val = float(rs[p])
        band_thr = peak_val + 0.5 * (median_std - peak_val)
        top = int(p)
        while top > 0 and rs[top - 1] <= band_thr:
            top -= 1
        bot = int(p)
        while bot < len(rs) - 1 and rs[bot + 1] <= band_thr:
            bot += 1
        centered.append((top + bot) // 2)
        # Estimate half-thickness; clip to a sane range
        half = max(2, min(60, (bot - top) // 2))
        half_thicknesses.append(int(half))

    dividers = [int(c + y1) for c in centered]
    thicknesses = list(half_thicknesses)

    # Sort + dedupe: centring can push neighbouring peaks onto the same / a
    # crossing position, leaving the lists unsorted with near-duplicates.
    if dividers:
        paired = sorted(zip(dividers, thicknesses), key=lambda t: t[0])
        dd_divs: list[int] = []
        dd_ths:  list[int] = []
        for d, t in paired:
            if dd_divs and d - dd_divs[-1] <= max(8, min_dist // 3):
                dd_ths[-1] = max(dd_ths[-1], t)   # merge → keep thicker band
                continue
            dd_divs.append(d)
            dd_ths.append(t)
        dividers, thicknesses = dd_divs, dd_ths

    # Box top and bottom serve as implicit outer dividers (no real listón
    # there, so half-thickness = 0 → no margin trimmed on that side).
    if not dividers or dividers[0] > y1 + min_dist:
        dividers.insert(0, y1)
        thicknesses.insert(0, 0)
    if dividers[-1] < y2 - min_dist:
        dividers.append(y2)
        thicknesses.append(0)

    # Cap at expected_max_channels + 1 outer rails — keep the most prominent
    max_divs = expected_max_channels + 1
    if len(dividers) > max_divs:
        inner = [(dividers[i], thicknesses[i], score[dividers[i] - y1])
                 for i in range(1, len(dividers) - 1)]
        inner.sort(key=lambda t: -t[2])     # by prominence desc
        keep_inner = inner[:max_divs - 2]
        keep_inner.sort(key=lambda t: t[0])
        dividers   = [dividers[0]]   + [d for d, _, _ in keep_inner] + [dividers[-1]]
        thicknesses = [thicknesses[0]] + [t for _, t, _ in keep_inner] + [thicknesses[-1]]

    return dividers, thicknesses


def _hough_row_votes(gray: np.ndarray, box_w: int, sigma: float) -> np.ndarray:
    """
    Per-row vote signal from long, near-horizontal Hough line segments.

    A real listón produces one or more long horizontal segments spanning a
    big fraction of the box width.  We accumulate, per Y row, the total
    length of near-horizontal segments passing through it.  Local rock-edge
    fragments are short and contribute little.
    """
    h, w = gray.shape
    votes = np.zeros(h, dtype=float)
    edges = cv2.Canny(gray, 50, 150)
    min_len = max(20, int(box_w * 0.45))   # at least ~45% of the width
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=min_len, maxLineGap=int(box_w * 0.1),
    )
    if lines is not None:
        for ln in lines:
            ax, ay, bx, by = ln[0]
            if abs(by - ay) <= 4:          # near-horizontal only
                yy = (ay + by) // 2
                if 0 <= yy < h:
                    votes[yy] += abs(bx - ax)
    return gaussian_filter1d(votes, sigma=sigma)


def _apply_periodicity_boost(score: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Softly boost the score at positions consistent with the dominant spacing.

    Uses autocorrelation of the score to estimate the period, then adds a
    gaussian comb aligned to the score's global maximum.  Weight is modest
    (0.2) so a wrong period estimate can't dominate the result.
    """
    n = len(score)
    sc = score - score.mean()
    acf = np.correlate(sc, sc, mode="full")[n - 1:]
    if acf.max() <= 0 or n < 2 * min_dist:
        return score
    acf = acf.copy()
    acf[:min_dist] = 0.0                    # ignore tiny lags
    period = int(np.argmax(acf))
    if period < min_dist:
        return score
    phase = int(np.argmax(score)) % period
    comb = np.zeros(n, dtype=float)
    for c in range(phase, n, period):
        comb[c] = 1.0
    comb = gaussian_filter1d(comb, sigma=max(2.0, period * 0.05))
    rng = comb.max() - comb.min()
    comb = (comb - comb.min()) / rng if rng > 1e-9 else comb
    return 0.8 * score + 0.2 * comb


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Row extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_rows(
    img: np.ndarray,
    box: np.ndarray,
    dividers: list[int],
    *,
    thicknesses: list[int] | None = None,
    min_channel_height: int = 30,
    require_rock_content: bool = True,
    liston_thickness: int = 0,
) -> list[Row]:
    """
    Slice the image into rows between consecutive dividers.

    Parameters
    ----------
    thicknesses : list[int] | None
        Half-thickness in pixels of each divider's physical listón.  Must be
        parallel to `dividers` if supplied.  When provided, each row strip
        is shrunk by `thicknesses[i]` on its top side and `thicknesses[i+1]`
        on its bottom side, skipping the actual wood/metal pixels.
    liston_thickness : int
        Fallback half-thickness used when `thicknesses` is not provided.
        Applied uniformly to both sides of every strip.
    require_rock_content : bool
        If True (default), strips that look like header cards, plain
        dividers, or ruler bars are filtered out.

    Returns a list of `Row` in reading order (top → bottom).
    """
    if len(dividers) < 2:
        return []

    if thicknesses is not None and len(thicknesses) != len(dividers):
        raise ValueError("thicknesses must match dividers in length")

    x1, _, x2, _ = _box_to_bbox(box)
    img_h = img.shape[0]
    rows: list[Row] = []
    idx = 0
    for i in range(len(dividers) - 1):
        y_top = int(dividers[i])
        y_bot = int(dividers[i + 1])
        th_top = thicknesses[i]     if thicknesses else liston_thickness
        th_bot = thicknesses[i + 1] if thicknesses else liston_thickness

        y_top_clip = max(0,     y_top + int(th_top))
        y_bot_clip = min(img_h, y_bot - int(th_bot))
        if y_bot_clip - y_top_clip < min_channel_height:
            continue
        strip = img[y_top_clip:y_bot_clip, x1:x2].copy()

        if require_rock_content and not _looks_like_rock(strip):
            continue

        rows.append(
            Row(
                img=strip,
                bbox=(x1, y_top_clip, x2, y_bot_clip),
                index=idx,
            )
        )
        idx += 1
    return rows


def _looks_like_rock(strip: np.ndarray) -> bool:
    """
    Heuristic: a strip contains rock content if its texture/colour pattern
    matches "many small fragments of medium-brightness coloured material"
    rather than "uniform divider", "mostly-white label card" or "ruler bar".

    Returns True if it looks like rocks.

    Rejection criteria
    ------------------
      A) Too uniform                  → divider / blank strip
      B) Almost completely white      → header / label card
      C) Almost completely black      → shadow / void
      D) Highly bimodal black-white   → ruler / checker bar
      E) Mostly-white + low colour    → label card with text (high edges
                                        but the background gives it away)
    """
    if strip.ndim != 3 or strip.size == 0:
        return False

    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)

    mean_v   = float(gray.mean())
    median_v = float(np.median(gray))   # robust against text-on-white skew
    std_v    = float(gray.std())
    sat_mean = float(hsv[:, :, 1].mean())

    # A · Uniform strip — std too low
    if std_v < 18:
        return False

    # B · Header card / label / paper sheet:
    #     dominant pixel value is very bright (mostly-white background),
    #     even if the mean is lower because of text & stickers.
    #     Median > 195 reliably catches white card backgrounds without
    #     rejecting light-grey granite (which sits around median ~150).
    if median_v > 195:
        return False

    # C · Shadow / void
    if mean_v < 25:
        return False

    # D · Ruler / checkerboard:
    #   • lots of extreme-dark + extreme-bright pixels (the checker pattern)
    #   • low colour saturation overall
    #   Real rocks rarely have >40 % of their pixels at the extremes.
    extreme_dark   = float((gray < 30).mean())
    extreme_bright = float((gray > 220).mean())
    if (extreme_dark + extreme_bright) > 0.40 and sat_mean < 60:
        return False

    # E · Edge density check — rocks have plenty of edges, but text on white
    #     card ALSO has edges. We've already handled the white-card case in (B),
    #     so we just need a minimum here to reject true blank strips.
    edges = cv2.Canny(gray, 50, 150)
    edge_frac = float(edges.mean()) / 255.0
    if edge_frac < 0.010:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _box_to_bbox(box: np.ndarray) -> tuple[int, int, int, int]:
    """Convert a 4-point polygon to its axis-aligned bounding box."""
    xs = box[:, 0]
    ys = box[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_detection_overlay(
    img: np.ndarray,
    box: np.ndarray,
    dividers: list[int],
    thicknesses: list[int] | None = None,
) -> np.ndarray:
    """
    Return a copy of `img` with the detected outline and dividers drawn on top.
    If `thicknesses` is given (parallel to `dividers`), each divider is
    rendered as a yellow band of that half-thickness with a centre line.
    Otherwise just thin yellow lines.
    """
    overlay = img.copy()
    x1, _, x2, _ = _box_to_bbox(box)

    # Box outline in green
    cv2.polylines(overlay, [box.reshape(-1, 1, 2)], True, (0, 220, 60), 4)

    if thicknesses and len(thicknesses) == len(dividers):
        # Faint yellow band + bright centre line
        band = overlay.copy()
        for y, th in zip(dividers, thicknesses):
            if th > 0:
                cv2.rectangle(
                    band, (x1, int(y) - int(th)), (x2, int(y) + int(th)),
                    (255, 230, 0), thickness=-1,
                )
        cv2.addWeighted(band, 0.25, overlay, 0.75, 0, overlay)
        for y in dividers:
            cv2.line(overlay, (x1, int(y)), (x2, int(y)), (255, 230, 0), 2)
    else:
        for y in dividers:
            cv2.line(overlay, (x1, int(y)), (x2, int(y)), (255, 230, 0), 2)

    return overlay
