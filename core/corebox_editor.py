"""
Manual editor for Photo Cores — unified canvas (box + dividers + thickness).
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from core.corebox import extract_rows
from core.divider_canvas import divider_canvas


def unified_editor(pc: dict, img: np.ndarray, key: str) -> bool:
    """
    One-canvas drag-and-drop editor for box + dividers + per-line thickness.

      • Drag green corners/edges of the box → resize it
      • Drag a yellow line → move it
      • Drag the ▲/▼ handles → adjust per-line thickness (half-band)
      • Click on empty area → add a new line
      • Double-click a line → mark for deletion (or remove if newly added)

    Parameters
    ----------
    pc   : photocore dict (mutated in place on apply)
    img  : working image, RGB numpy array

    Returns
    -------
    bool — True if the user applied changes this rerun (caller should
    persist + rerun); False otherwise.
    """
    result = divider_canvas(
        img,
        pc["dividers"],
        box=pc["box"],
        thicknesses=pc.get("thicknesses"),
        max_canvas_width=1100,
        key=f"unifcanvas_{key}",
    )

    if not result or not isinstance(result, dict):
        return False

    ts = result.get("timestamp", 0)
    if pc.get("_last_canvas_ts") == ts:
        return False         # already processed in a previous rerun
    pc["_last_canvas_ts"] = ts

    action = result.get("action")
    if action == "apply":
        _apply_changes(pc, img, result)
        return True
    if action == "reset":
        pc["params_key"] = None
        pc["source"] = "auto"
        pc.pop("_last_canvas_ts", None)
        return True
    return False


def _apply_changes(pc: dict, img: np.ndarray, result: dict) -> None:
    # ── Box ───────────────────────────────────────────────────────────────────
    nb = result.get("box")
    if nb and nb.get("x2", 0) > nb.get("x1", 0) and nb.get("y2", 0) > nb.get("y1", 0):
        x1, y1, x2, y2 = nb["x1"], nb["y1"], nb["x2"], nb["y2"]
        pc["box"] = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32,
        )

    # ── Dividers + thicknesses (parallel lists) ──────────────────────────────
    new_ys = [int(y) for y in (result.get("dividers") or [])]
    new_ts = [int(t) for t in (result.get("thicknesses") or [])]
    paired = sorted(
        zip(new_ys, new_ts if len(new_ts) == len(new_ys) else [0] * len(new_ys)),
        key=lambda p: p[0],
    )
    dedup: list[tuple[int, int]] = []
    for y, t in paired:
        if not dedup or y - dedup[-1][0] > 8:
            dedup.append((y, t))
    pc["dividers"]    = [y for y, _ in dedup]
    pc["thicknesses"] = [t for _, t in dedup]

    # ── Re-extract rows with the updated per-line thicknesses ────────────────
    pc["rows"] = extract_rows(
        img, pc["box"], pc["dividers"],
        thicknesses=pc["thicknesses"],
    )
    pc["rows_raw"] = extract_rows(
        img, pc["box"], pc["dividers"],
        thicknesses=pc["thicknesses"],
        require_rock_content=False,
    )
    pc["source"] = "manual"
