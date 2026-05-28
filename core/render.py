import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# ── Palette ──────────────────────────────────────────────────────────────────
ZONE_RGB = {
    "intact":    ( 34, 197,  94),   # green
    "fractured": (234, 179,   8),   # yellow/amber
    "rubble":    (239,  68,  68),   # red
    "wood":      (160, 100,  40),   # warm brown
}
ZONE_NORM = {k: tuple(v / 255 for v in c) for k, c in ZONE_RGB.items()}


# ── Zone label helpers ────────────────────────────────────────────────────────

def zone_bar_label(z):
    """
    Short text label shown inside the zone bar.
      intact    → I
      rubble    → R
      wood      → W
      fractured → MJ  if mechanical (p_mec > p_nat)
                  J   if natural    (p_nat >= p_mec)
    """
    lbl = z["label"]
    if lbl == "fractured":
        return "MJ" if z.get("p_mec", 0) > z.get("p_nat", 0) else "J"
    return lbl[0].upper()


# ── Public helpers ────────────────────────────────────────────────────────────

def render_zone_bar(img_width, zones, bar_height=44):
    """Return (bar_height × img_width × 3) uint8 RGB array."""
    bar = np.full((bar_height, img_width, 3), 180, dtype=np.uint8)
    for z in zones:
        bar[:, z["x_start"]: z["x_end"]] = ZONE_RGB[z["label"]]
    return bar


def render_debug_views(img, mask, windows, all_feats, raw_labels, smoothed_labels, zones):
    """
    Return an ordered dict of matplotlib Figures, one per debug step.
    Close them after use to free memory.
    """
    return {
        "raw_row":          _fig_raw_row(img),
        "rock_mask":        _fig_rock_mask(img, mask),
        "sliding_windows":  _fig_windows(img, windows),
        "feature_profiles": _fig_feature_profiles(all_feats, windows, img.shape[1]),
        "raw_labels":       _fig_label_bar(raw_labels,       windows, img.shape[1], "Raw Labels (before smoothing)"),
        "smoothed_labels":  _fig_label_bar(smoothed_labels,  windows, img.shape[1], "Smoothed Labels"),
        "final_zones":      _fig_final_zones(img, zones),
    }


def fig_to_pil(fig):
    """Convert a matplotlib Figure to a PIL Image (for st.image)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


# ── Internal figure builders ─────────────────────────────────────────────────

def _fig_raw_row(img):
    fig, ax = plt.subplots(figsize=_img_figsize(img))
    ax.imshow(img)
    ax.set_title("1 · Raw Row Input", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout(pad=0.4)
    return fig


def _fig_rock_mask(img, mask):
    from core.mask import overlay_mask_on_image
    fw, fh = _img_figsize(img)
    fig, axes = plt.subplots(1, 2, figsize=(fw, fh))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay_mask_on_image(img, mask), interpolation="nearest")
    axes[1].set_title("Rock Mask overlay  (green = rock detected)")
    axes[1].axis("off")
    fig.suptitle("2 · Rock Mask", fontsize=11, fontweight="bold")
    plt.tight_layout(pad=0.4)
    return fig


def _fig_windows(img, windows, max_shown=30):
    fig, ax = plt.subplots(figsize=_img_figsize(img))
    ax.imshow(img)
    step = max(1, len(windows) // max_shown)
    for i, (x1, x2) in enumerate(windows[::step]):
        color = "cyan" if i % 2 == 0 else "yellow"
        rect = mpatches.Rectangle(
            (x1, 0), x2 - x1, img.shape[0],
            linewidth=0.6, edgecolor=color, facecolor=color, alpha=0.18
        )
        ax.add_patch(rect)
    ax.set_title(
        f"3 · Sliding Windows  (showing every {step} of {len(windows)} total)",
        fontsize=11, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout(pad=0.4)
    return fig


def _img_figsize(img, display_w=12.0, extra_h=0.4):
    """Return (width, height) in inches that preserves the image aspect ratio."""
    h, w = img.shape[:2]
    display_h = max(0.5, display_w * h / w) + extra_h
    return (display_w, display_h)


def _fig_feature_profiles(all_feats, windows, img_width):
    centers = np.array([(x1 + x2) / 2 for x1, x2 in windows])

    FEATURES = {
        "crack_column_fraction":    ("tab:red",    "solid"),    # primary crack signal
        "brightness_cv":            ("tomato",     "dashed"),   # contrast variation
        "local_fragmentation":      ("firebrick",  "dotted"),   # aggregate
        "rock_occupancy":           ("tab:blue",   "solid"),
        "largest_component_ratio":  ("tab:green",  "solid"),
        "width_stability":          ("tab:purple", "dashed"),
        "longitudinal_continuity":  ("tab:orange", "dashed"),
    }

    fig, ax = plt.subplots(figsize=(12, 3.5))
    for fname, (color, ls) in FEATURES.items():
        vals = [f.get(fname, 0.0) for f in all_feats]
        ax.plot(centers, vals, label=fname, color=color,
                linestyle=ls, linewidth=1.6, alpha=0.85)

    ax.set_xlim(0, img_width)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("Pixel position (x)")
    ax.set_ylabel("Feature value  [0 – 1]")
    ax.set_title("4 · Feature Profiles Along Row", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.4)
    return fig


def _fig_label_bar(labels, windows, img_width, title):
    bar_h = 60
    bar = np.full((bar_h, img_width, 3), 180, dtype=np.uint8)
    for (x1, x2), lbl in zip(windows, labels):
        bar[:, x1:x2] = ZONE_RGB.get(lbl, (180, 180, 180))

    fig, ax = plt.subplots(figsize=(12, 1.4))
    ax.imshow(bar, aspect="auto")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    _add_legend(ax)
    plt.tight_layout(pad=0.3)
    return fig


def _fig_final_zones(img, zones):
    h, w = img.shape[:2]
    bar_h = 50
    bar   = render_zone_bar(w, zones, bar_h)
    combined = np.vstack([img, bar])

    display_w = 12.0
    display_h = max(0.8, display_w * combined.shape[0] / w) + 0.3
    fig, ax = plt.subplots(figsize=(display_w, display_h))
    ax.imshow(combined)   # aspect="equal" por defecto — sin distorsión

    # Label each zone in the bar strip
    for z in zones:
        cx  = (z["x_start"] + z["x_end"]) / 2
        cy  = h + bar_h / 2
        lbl = zone_bar_label(z)
        ax.text(cx, cy, lbl,
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

    ax.set_title("7 · Final Zones", fontsize=11, fontweight="bold")
    ax.axis("off")
    _add_legend(ax)
    plt.tight_layout(pad=0.4)
    return fig


def _add_legend(ax):
    patches = [
        mpatches.Patch(color=ZONE_NORM["intact"],    label="I — Intact"),
        mpatches.Patch(color=ZONE_NORM["fractured"], label="J — Natural Joint"),
        mpatches.Patch(color=ZONE_NORM["fractured"], label="MJ — Mechanical Joint",
                       linestyle="--", linewidth=1.5),
        mpatches.Patch(color=ZONE_NORM["rubble"],    label="R — Rubble"),
        mpatches.Patch(color=ZONE_NORM["wood"],      label="W — Wood block"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              framealpha=0.7, handlelength=1.2)


# ── Ruler helpers (public) ────────────────────────────────────────────────────

def render_row_with_ruler(img, img_width, row_length_cm=None):
    """
    Show the original row image with a cm / px ruler on the X axis.
    Preserves the image's natural aspect ratio — no stretching.
    """
    h, w = img.shape[:2]
    display_w = 12.0
    # Keep the natural aspect ratio; add 0.55 in for the ruler below
    display_h = max(0.4, display_w * h / w) + 0.55

    fig, ax = plt.subplots(figsize=(display_w, display_h))
    ax.imshow(img)                          # default aspect="equal", no extent
    _apply_ruler(ax, w, row_length_cm)
    ax.set_title("Original row", fontsize=11)
    ax.set_yticks([])
    plt.tight_layout(pad=0.4)
    return fig


def render_bar_with_ruler(img_width, zones, row_length_cm=None, bar_height=48):
    """
    Show the zone colour bar with a cm / px ruler on the X axis.
    """
    bar = render_zone_bar(img_width, zones, bar_height)

    fig, ax = plt.subplots(figsize=(12, 1.6))
    ax.imshow(bar, aspect="auto")           # bar can stretch freely — it's just colour

    # Zone labels centred in each band
    for z in zones:
        cx  = (z["x_start"] + z["x_end"]) / 2
        lbl = zone_bar_label(z)
        ax.text(cx, bar_height / 2, lbl,
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", alpha=0.85)

    _apply_ruler(ax, img_width, row_length_cm)
    ax.set_title("Zone classification", fontsize=11)
    ax.set_yticks([])
    _add_legend(ax)
    plt.tight_layout(pad=0.4)
    return fig


def _apply_ruler(ax, img_width, row_length_cm):
    """
    Add a readable X-axis ruler.
    - If row_length_cm is given: ticks every N cm (auto-spaced).
    - Otherwise: ticks every ~10 % of width in pixels.
    """
    if row_length_cm:
        # Pick a round tick interval: aim for ~8-12 ticks
        raw_step = row_length_cm / 10
        nice_steps = [0.5, 1, 2, 5, 10, 20, 25, 50]
        step_cm = min(nice_steps, key=lambda s: abs(s - raw_step))

        ticks_cm = np.arange(0, row_length_cm + step_cm * 0.01, step_cm)
        ticks_px = ticks_cm / row_length_cm * img_width

        ax.set_xticks(ticks_px)
        ax.set_xticklabels([f"{v:.1f}" for v in ticks_cm], fontsize=7)
        ax.set_xlabel("Position (cm)", fontsize=8)
    else:
        # Pixel ticks every ~10 %
        step_px = max(1, round(img_width / 10 / 50) * 50)
        ticks_px = np.arange(0, img_width + 1, step_px)
        ax.set_xticks(ticks_px)
        ax.set_xticklabels([str(int(v)) for v in ticks_px], fontsize=7)
        ax.set_xlabel("Position (px)", fontsize=8)

    ax.set_xlim(0, img_width)
    ax.tick_params(axis="x", direction="out", length=3)
