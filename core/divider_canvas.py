"""
Custom Streamlit component for drag-and-drop divider editing.

Renders an HTML5 canvas with the photo as background and each divider as a
horizontal yellow line.  User can:
  • drag a line up/down to move it
  • click on empty area to add a new line
  • double-click on a line to mark it for deletion (or remove it if newly added)
  • press "Aplicar" to send the new divider list back to Python

Returns a dict like
    {"action": "apply", "dividers": [..], "timestamp": ms}
or `None` if the user hasn't pressed Apply yet.

The frontend is a single static `index.html` — no npm/build step needed.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image as PILImage


_FRONTEND_DIR = Path(__file__).parent / "divider_canvas_html"
_component = components.declare_component(
    "divider_canvas",
    path=str(_FRONTEND_DIR),
)


def _image_to_data_url(img: np.ndarray, max_width: int = 900,
                       jpeg_quality: int = 72) -> tuple[str, int, int]:
    """
    Convert a numpy RGB image to a JPEG data URL, downscaled so its width
    doesn't exceed `max_width` (keeps the payload small).
    Returns (data_url, original_w, original_h).
    """
    h, w = img.shape[:2]
    pil = PILImage.fromarray(img) if img.dtype == np.uint8 else PILImage.fromarray(
        img.astype(np.uint8)
    )
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if w > max_width:
        new_h = int(h * max_width / w)
        pil = pil.resize((max_width, new_h), PILImage.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{data}", w, h


def divider_canvas(
    img: np.ndarray,
    dividers: list[int],
    box=None,
    thicknesses: list[int] | None = None,
    *,
    max_canvas_width: int = 900,
    key: str | None = None,
) -> dict | None:
    """
    Render the unified drag-and-drop editor (box + dividers + thicknesses).

    Parameters
    ----------
    img           : RGB numpy image (the photo, processed resolution)
    dividers      : list of Y coordinates (image space)
    box           : 4-point polygon or None → defaults to full image
    thicknesses   : list of half-thickness px per divider, parallel to dividers.
                    None → each line gets a default half-thickness.
    max_canvas_width : px — canvas display width cap
    key           : streamlit widget key

    Returns
    -------
    None until the user clicks a toolbar button, then one of:
        {"action": "apply", "box": {x1,y1,x2,y2}, "dividers": [int,...],
         "thicknesses": [int,...], "timestamp": ms}
        {"action": "reset", "timestamp": ms}
    All coordinates are in IMAGE space (matching the `img` passed in).
    """
    if img is None or img.size == 0:
        return None
    image_url, w, h = _image_to_data_url(img, max_width=max_canvas_width)

    # Convert box polygon → axis-aligned dict; None → fall back in JS
    box_dict = None
    if box is not None and len(box) >= 3:
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        box_dict = {"x1": min(xs), "y1": min(ys),
                    "x2": max(xs), "y2": max(ys)}

    th_list = None
    if thicknesses is not None and len(thicknesses) == len(dividers):
        th_list = [int(t) for t in thicknesses]

    return _component(
        image_url=image_url,
        image_w=w,
        image_h=h,
        box=box_dict,
        dividers=[int(y) for y in dividers],
        thicknesses=th_list,
        max_w=max_canvas_width,
        default=None,
        key=key,
    )
