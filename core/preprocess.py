import cv2
import numpy as np


def load_row_image(source):
    """Load image from a file path, bytes, or Streamlit UploadedFile."""
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Cannot load image: {source}")
    else:
        data = source.read() if hasattr(source, "read") else source
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to_width(img, target_width):
    """Resize image to target_width preserving aspect ratio."""
    h, w = img.shape[:2]
    if w == target_width:
        return img
    target_height = max(1, int(h * target_width / w))
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)


def preprocess_row(img, apply_clahe=True, crop_margins=True):
    """Gentle preprocessing: margin crop + optional CLAHE on L channel."""
    result = img.copy()

    if crop_margins:
        result = _crop_empty_margins(result)

    if apply_clahe and result.size > 0:
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result


def _crop_empty_margins(img, bright_threshold=245, dark_threshold=12):
    """Remove nearly-white or nearly-black border bands."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    is_rock = (gray > dark_threshold) & (gray < bright_threshold)

    if not is_rock.any():
        return img

    rows = np.where(is_rock.any(axis=1))[0]
    cols = np.where(is_rock.any(axis=0))[0]

    pad = 4
    y1 = max(0, rows[0] - pad)
    y2 = min(h, rows[-1] + 1 + pad)
    x1 = max(0, cols[0] - pad)
    x2 = min(w, cols[-1] + 1 + pad)

    return img[y1:y2, x1:x2]
