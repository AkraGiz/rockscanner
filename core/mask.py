import cv2
import numpy as np
from scipy import ndimage


def build_rock_mask(img):
    """
    Return binary mask: 255 = rock, 0 = background.
    Adapts to bright (white tray) or dark backgrounds automatically.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    median_val = float(np.median(gray))

    if median_val > 140:
        mask = _mask_bright_background(gray)
    else:
        mask = _mask_dark_background(gray)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Fill internal holes so solid rock reads as solid
    mask = ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255

    return mask


def _mask_bright_background(gray):
    """Rock is darker than a bright (white/cream) background."""
    # Otsu finds the natural break between background and rock
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Also exclude very bright pixels (tray/paper) regardless of Otsu
    bright_cutoff = max(otsu_val, 200)
    return ((gray < bright_cutoff)).astype(np.uint8) * 255


def _mask_dark_background(gray):
    """Rock is lighter than a dark background (dark tray)."""
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (gray > otsu_val).astype(np.uint8) * 255


def overlay_mask_on_image(img, mask, alpha=0.35, color=(40, 220, 80)):
    """Return RGB image with the rock mask tinted in `color`."""
    overlay = img.copy().astype(float)
    rock = mask > 0
    for c, val in enumerate(color):
        overlay[:, :, c][rock] = overlay[:, :, c][rock] * (1 - alpha) + val * alpha
    return overlay.clip(0, 255).astype(np.uint8)
