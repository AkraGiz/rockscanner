import numpy as np


def compute_sliding_windows(img_width, window_width_ratio=0.05, stride_ratio=0.02):
    """
    Generate (x_start, x_end) sliding windows along the horizontal axis.

    Args:
        img_width: total width of the image in pixels
        window_width_ratio: window width as fraction of img_width
        stride_ratio: stride as fraction of img_width

    Returns:
        List of (x_start, x_end) tuples (non-overlapping coverage guaranteed)
    """
    win_w  = max(8, int(img_width * window_width_ratio))
    stride = max(1, int(img_width * stride_ratio))

    windows = []
    x = 0
    while x + win_w <= img_width:
        windows.append((x, x + win_w))
        x += stride

    # Ensure the trailing edge is fully covered
    if windows and windows[-1][1] < img_width:
        last_start = max(0, img_width - win_w)
        windows.append((last_start, img_width))

    return windows
