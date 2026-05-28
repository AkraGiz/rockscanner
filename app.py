"""
Photo Cores — Streamlit home page.

Multi-image core box analyzer.  Detects box outline + horizontal dividers
on each uploaded photo, lets the user adjust the detection with a custom
drag-and-drop canvas, and persists everything to disk under
`data/photocores/<hash>/` so that work survives browser reloads.

Each extracted row can be sent to the **Row Analyzer** page to run the
full per-row fracture / zone classification pipeline.
"""

from __future__ import annotations

import io
import logging
import hashlib
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Silence Streamlit's harmless media-cache "Missing file" log spam
for _noisy in (
    "streamlit.runtime.memory_media_file_storage",
    "streamlit.web.server.media_file_handler",
):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

from core.corebox import (
    detect_box_outline,
    detect_horizontal_dividers,
    extract_rows,
    draw_detection_overlay,
    detect_skew_angle,
    deskew_image,
)
from core.corebox_editor import unified_editor
from core import storage


# Path to the row analyzer page (used by st.switch_page)
ROW_ANALYZER_PAGE = "pages/2_📊_Row_Analyzer.py"


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Photo Cores", page_icon="📦", layout="wide")
st.title("📦 Photo Cores — Box-level analysis")
st.caption(
    "Upload photos of drill-core boxes.  The auto-detection finds the box "
    "outline and the dividers between rows; you can adjust everything with "
    "drag&drop.  All state is persisted on disk under `data/photocores/` and "
    "survives browser reloads.  Send any extracted row to the Row Analyzer "
    "for the full fracture / zone classification."
)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state utilities
# ─────────────────────────────────────────────────────────────────────────────

def _img_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()[:12]


def _params_key(sensitivity, max_channels, deskew, max_long_side, sig_flags) -> str:
    """Detection cache key — invalidates auto-detection when sliders/toggles move."""
    flags = "".join("1" if f else "0" for f in sig_flags)
    return (f"s{sensitivity:.2f}_m{max_channels}_d{int(deskew)}"
            f"_L{max_long_side}_f{flags}")


def _downscale_to_long_side(img: np.ndarray, max_long_side) -> np.ndarray:
    if max_long_side in (None, "Original"):
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= int(max_long_side):
        return img
    s = int(max_long_side) / long_side
    return cv2.resize(
        img, (max(1, int(w * s)), max(1, int(h * s))),
        interpolation=cv2.INTER_AREA,
    )


def _get_img(pc: dict) -> np.ndarray:
    """Lazy-load the processed image from disk into the pc cache."""
    if pc.get("_img_cache") is None:
        pc["_img_cache"] = storage.load_processed_image(pc["key"])
    return pc["_img_cache"]


def _persist(pc: dict) -> None:
    """Write the current photocore state to disk (metadata + overlay + rows)."""
    key = pc["key"]
    img = _get_img(pc)
    if img is None:
        return
    overlay = draw_detection_overlay(
        img, pc["box"], pc["dividers"], thicknesses=pc.get("thicknesses"),
    )
    storage.save_overlay(key, overlay)
    rows = pc.get("rows") or []
    storage.save_rows(key, [r.img for r in rows])
    storage.save_metadata(key, _to_metadata(pc))


def _to_metadata(pc: dict) -> dict:
    box_list = pc["box"].tolist() if hasattr(pc["box"], "tolist") else pc["box"]
    return {
        "name":             pc["name"],
        "hash":             pc["key"],
        "original_shape":   pc.get("original_shape"),
        "max_long_side":    pc.get("max_long_side"),
        "deskew_on":        pc.get("deskew_on", False),
        "skew_angle":       pc.get("skew_angle", 0.0),
        "box":              box_list,
        "dividers":         pc["dividers"],
        "thicknesses":      pc.get("thicknesses", []),
        "params_key":       pc.get("params_key"),
        "source":           pc.get("source", "auto"),
        "rows": [
            {"index": r.index, "bbox": list(r.bbox)}
            for r in (pc.get("rows") or [])
        ],
    }


def _from_metadata(meta: dict, key: str) -> dict:
    """Reconstruct a pc dict from metadata (without loading the image yet)."""
    box = np.array(meta["box"], dtype=np.int32) if meta.get("box") else None
    return {
        "key":              key,
        "name":             meta.get("name", key),
        "original_shape":   meta.get("original_shape"),
        "max_long_side":    meta.get("max_long_side"),
        "deskew_on":        meta.get("deskew_on", False),
        "skew_angle":       meta.get("skew_angle", 0.0),
        "box":              box,
        "dividers":         meta.get("dividers", []),
        "thicknesses":      meta.get("thicknesses", []),
        "params_key":       meta.get("params_key"),
        "source":           meta.get("source", "auto"),
        "rows":             [],
        "_img_cache":       None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Restore existing photocores from disk on first load
# ─────────────────────────────────────────────────────────────────────────────

if "photocores" not in st.session_state:
    st.session_state["photocores"] = {}
    for k in storage.list_photocore_keys():
        meta = storage.load_metadata(k)
        if meta:
            st.session_state["photocores"][k] = _from_metadata(meta, k)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Detection parameters")
    sensitivity = st.slider(
        "Sensitivity", 0.0, 1.0, 0.30, 0.05,
        help=(
            "How many dividers to count.\n"
            "• Low (~0.2)  → catches THIN dividers (metal cucharas)\n"
            "• High (~0.7) → only THICK dividers (classic wood boxes)\n"
            "If too many / too few rows show up, move this slider."
        ),
    )
    max_channels = st.slider("Max channels per box", 5, 25, 15, 1)

    with st.expander("🔬 Detection signals", expanded=False):
        st.caption(
            "Each signal contributes to the divider score. Toggle them to "
            "fine-tune precision for tricky boxes."
        )
        use_variance   = st.checkbox(
            "Variance (uniform strip)", value=True,
            help="Low horizontal variance = uniform divider. The base signal.",
        )
        use_brightness = st.checkbox(
            "Brightness transitions", value=True,
            help="Sharp change in mean brightness = content-type boundary.",
        )
        use_sobel      = st.checkbox(
            "Horizontal edge magnitude", value=True,
            help="Strong horizontal edges per row (Sobel Y).",
        )
        use_continuity = st.checkbox(
            "Edge continuity (full width)", value=True,
            help="Fraction of the width covered by a strong edge. A real "
                 "listón runs edge-to-edge; inter-rock shadows are local. "
                 "The most precise signal.",
        )
        use_hough      = st.checkbox(
            "Hough line votes", value=False,
            help="Votes from long near-horizontal line segments. Good for "
                 "boxes with very clean, straight listones.",
        )
        use_periodicity = st.checkbox(
            "Periodicity boost (equispaced rows)", value=False,
            help="Softly boosts positions consistent with the dominant row "
                 "spacing. Only helps when rows are regularly spaced.",
        )

    st.divider()
    max_long_side = st.select_slider(
        "📐 Max size (long edge, px)",
        options=[800, 1080, 1200, 1500, 2000, 2500, "Original"],
        value=1500,
        help=(
            "Uploaded photos are rescaled to this size on their long edge "
            "before any processing.  1500 px is plenty for divider detection "
            "and keeps the app snappy.  4000+ px images can stall the browser."
        ),
    )
    auto_deskew = st.checkbox(
        "🧭 Auto-deskew image", value=True,
        help="Rotate the image so horizontal features stay horizontal.",
    )

    st.divider()
    show_rejected = st.checkbox(
        "Show rejected rows", value=False,
        help="Useful to see what strips were discarded as header/ruler/empty.",
    )

    # ── Saved photocores ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🗂 Saved boxes")
    saved_keys = list(st.session_state["photocores"].keys())
    if saved_keys:
        for k in saved_keys:
            pc_meta = st.session_state["photocores"][k]
            size_mb = storage.folder_size_mb(k)
            col_a, col_b = st.columns([3, 1])
            col_a.caption(f"📷 {pc_meta['name'][:30]}  ({size_mb:.1f} MB)")
            if col_b.button("🗑", key=f"del_pc_{k}",
                            help="Delete this box from disk"):
                storage.delete_photocore(k)
                st.session_state["photocores"].pop(k, None)
                st.rerun()
        if st.button("🗑 Delete all", key="del_all_pcs"):
            n = storage.delete_all()
            st.session_state["photocores"].clear()
            st.toast(f"Deleted {n} boxes")
            st.rerun()
    else:
        st.caption("No saved boxes yet.")


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload one or more box photos",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Process newly uploaded files (or re-process when params changed)
# ─────────────────────────────────────────────────────────────────────────────

sig_flags = (use_variance, use_brightness, use_sobel,
             use_continuity, use_hough, use_periodicity)
params_key = _params_key(sensitivity, max_channels, auto_deskew,
                         max_long_side, sig_flags)

if uploaded:
    for file in uploaded:
        raw = file.read()
        key = _img_hash(raw)
        pc = st.session_state["photocores"].get(key)

        # ── First time: load + downscale ──────────────────────────────────────
        if pc is None:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            full = np.array(pil)
            scaled = _downscale_to_long_side(full, max_long_side)
            pc = {
                "key":            key,
                "name":           file.name,
                "original_shape": (full.shape[1], full.shape[0]),
                "max_long_side":  max_long_side,
                "deskew_on":      None,
                "skew_angle":     0.0,
                "box":            None,
                "dividers":       [],
                "thicknesses":    [],
                "params_key":     None,
                "source":         "auto",
                "rows":           [],
                "_img_cache":     scaled,
            }
            storage.save_processed_image(key, scaled)
            st.session_state["photocores"][key] = pc
            del full

        # ── Reprocess from raw if max_long_side changed ──────────────────────
        if pc.get("max_long_side") != max_long_side:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            full = np.array(pil)
            scaled = _downscale_to_long_side(full, max_long_side)
            pc["_img_cache"]    = scaled
            pc["max_long_side"] = max_long_side
            pc["deskew_on"]     = None
            pc["params_key"]    = None
            pc["source"]        = "auto"
            storage.save_processed_image(key, scaled)
            del full

        # ── Apply / re-apply deskew if the toggle differs ────────────────────
        if pc.get("deskew_on") != auto_deskew:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            base = _downscale_to_long_side(np.array(pil), max_long_side)
            if auto_deskew:
                angle = detect_skew_angle(base)
                rotated = deskew_image(base, angle) if abs(angle) >= 0.3 else base
                pc["_img_cache"] = rotated
                pc["skew_angle"] = angle
                storage.save_processed_image(key, rotated)
            else:
                pc["_img_cache"] = base
                pc["skew_angle"] = 0.0
                storage.save_processed_image(key, base)
            pc["deskew_on"]  = auto_deskew
            pc["params_key"] = None
            pc["source"]     = "auto"
            del pil

        # ── Auto-detect if needed ────────────────────────────────────────────
        needs_auto = (pc.get("params_key") != params_key
                      and pc.get("source") != "manual")
        if needs_auto:
            img = _get_img(pc)
            with st.spinner(f"Detecting structure in {file.name}…"):
                box = detect_box_outline(img)
                divs, thicks = detect_horizontal_dividers(
                    img, box,
                    expected_max_channels=max_channels,
                    sensitivity=sensitivity,
                    use_variance=use_variance,
                    use_brightness=use_brightness,
                    use_sobel=use_sobel,
                    use_continuity=use_continuity,
                    use_hough=use_hough,
                    use_periodicity=use_periodicity,
                )
                rows = extract_rows(img, box, divs, thicknesses=thicks)
                rows_raw = extract_rows(
                    img, box, divs, thicknesses=thicks, require_rock_content=False,
                )
            pc.update({
                "box":         box,
                "dividers":    divs,
                "thicknesses": thicks,
                "rows":        rows,
                "rows_raw":    rows_raw,
                "source":      "auto",
                "params_key":  params_key,
            })
            _persist(pc)


# ─────────────────────────────────────────────────────────────────────────────
# Render each photocore
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state["photocores"]:
    st.info("Upload at least one box photo to begin.")
    st.stop()

for key, pc in list(st.session_state["photocores"].items()):
    st.divider()
    st.header(f"📦 {pc['name']}")

    img = _get_img(pc)
    if img is None or pc["box"] is None:
        st.warning("Image not available on disk. Delete this box and upload it again.")
        continue

    box  = pc["box"]
    divs = pc["dividers"]
    ths  = pc.get("thicknesses") or []
    rows = pc["rows"] if pc.get("rows") else extract_rows(
        img, box, divs, thicknesses=ths,
    )
    pc["rows"]     = rows
    pc["rows_raw"] = pc.get("rows_raw") or extract_rows(
        img, box, divs, thicknesses=ths, require_rock_content=False,
    )

    col_meta, col_img = st.columns([1, 3])
    with col_meta:
        rows_raw_count = len(pc.get("rows_raw") or rows)
        st.metric("Valid rows (with rock)", len(rows))
        if rows_raw_count != len(rows):
            st.caption(f"({rows_raw_count - len(rows)} discarded)")
        ih, iw = img.shape[:2]
        st.metric("Processed", f"{iw}×{ih} px")
        orig = pc.get("original_shape")
        if orig and tuple(orig) != (iw, ih):
            st.caption(f"(original {orig[0]}×{orig[1]}, downscaled on upload)")
        skew = pc.get("skew_angle", 0.0)
        if pc.get("deskew_on") and abs(skew) >= 0.3:
            st.caption(f"🧭 Deskewed {skew:+.1f}°")
        st.caption(f"Source: {pc['source']}  ·  sens={sensitivity:.2f}")
        st.caption(f"📁 `data/photocores/{key}/`")

    with col_img:
        ov_path = storage.overlay_path(key)
        if not ov_path.exists():
            overlay = draw_detection_overlay(img, box, divs, thicknesses=ths)
            storage.save_overlay(key, overlay)
        st.image(
            str(ov_path),
            caption="Current detection (green box · yellow dividers, band = thickness)",
            width="stretch",
        )

    # ── Editor (button-gated to avoid mounting the iframe by default) ────────
    badge      = "🤖 auto" if pc["source"] == "auto" else "✏️ manual"
    open_state = f"editor_open_{key}"
    if not st.session_state.get(open_state):
        if st.button(f"✏️ Edit box and dividers ({badge})", key=f"open_editor_{key}"):
            st.session_state[open_state] = True
            st.rerun()
    else:
        if st.button("✕ Close editor", key=f"close_editor_{key}"):
            st.session_state[open_state] = False
            st.rerun()
        changed = unified_editor(pc, img=img, key=key)
        if changed:
            _persist(pc)
            st.rerun()

    # ── Row strips (lazy expander, served from disk) ─────────────────────────
    with st.expander(f"📋 Extracted rows ({len(rows)})", expanded=False):
        st.caption(
            "Each row is a strip ready for analysis.  Click **▶ Analyze** to "
            f"send it to the Row Analyzer page.  Stored at "
            f"`data/photocores/{key}/rows/`."
        )
        display_rows = (pc.get("rows_raw") or rows) if show_rejected else rows
        rock_keys = {(r.bbox[1], r.bbox[3]) for r in rows}

        for row in display_rows:
            is_rock = (row.bbox[1], row.bbox[3]) in rock_keys
            row_col1, row_col2 = st.columns([4, 1])
            with row_col1:
                tag = "" if is_rock else "  ❌ (rejected)"
                rp = storage.row_path(key, row.index)
                src = str(rp) if rp.exists() else row.img
                st.image(
                    src,
                    caption=f"Row {row.index + 1}{tag}  ·  bbox {row.bbox}  ·  "
                            f"{row.img.shape[1]}×{row.img.shape[0]} px",
                    width="stretch",
                )
            with row_col2:
                if is_rock and st.button(
                    "▶ Analyze",
                    key=f"analyze_{key}_{row.index}",
                    type="primary",
                ):
                    st.session_state["staged_row"] = {
                        "img":    row.img,
                        "source": f"{pc['name']} · row {row.index + 1}",
                    }
                    st.switch_page(ROW_ANALYZER_PAGE)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "💾 Disk persistence enabled.  Boxes, row strips and metadata live under "
    "`data/photocores/` and are restored automatically on page load."
)
