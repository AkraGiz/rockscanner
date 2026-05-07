"""
PDF report generator for Rock Core Row Zonifier.
Uses reportlab — no browser printing required.
"""

import io
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from core.render import render_row_with_ruler, render_bar_with_ruler, render_debug_views
from core.fracture_signals import fracture_origin


_LABEL_ESP = {
    "intact":    "Intacto",
    "fractured": "Fracturado",
    "rubble":    "Rubble",
    "wood":      "Madera",
}
_LABEL_BG = {
    "Intacto":    colors.HexColor("#d5f5e3"),
    "Fracturado": colors.HexColor("#fef3cd"),
    "Rubble":     colors.HexColor("#fde8e8"),
    "Madera":     colors.HexColor("#f5e6d3"),
}

PAGE      = landscape(A4)
PW, PH    = PAGE          # ~841 × 595 pt
MARGIN    = 18 * mm
USABLE_W  = PW - 2 * MARGIN
USABLE_H  = PH - 26 * mm  # top + bottom margins


def _fig_to_buf(fig, dpi=120):
    """Save figure to BytesIO and return (buf, aspect_ratio h/w)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    pil = PILImage.open(buf)
    aspect = pil.size[1] / pil.size[0]
    buf.seek(0)
    return buf, aspect


def _rl_image(buf, aspect, max_h=None):
    """
    RLImage always rendered at USABLE_W (full page width).
    If max_h is set and the computed height exceeds it, the image is scaled
    down proportionally — both width and height — to fit.
    This preserves alignment between images on the summary page (same width)
    while preventing very tall debug figures from overflowing the page.
    """
    w = USABLE_W
    h = w * aspect
    if max_h and h > max_h:
        scale = max_h / h
        w *= scale
        h = max_h
    return RLImage(buf, width=w, height=h)


def _make_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "rz_title", parent=styles["Title"],
            fontSize=13, spaceAfter=3, textColor=colors.HexColor("#1a252f"),
        ),
        "h2": ParagraphStyle(
            "rz_h2", parent=styles["Heading2"],
            fontSize=9, spaceBefore=8, spaceAfter=2,
            textColor=colors.HexColor("#2c3e50"),
        ),
        "h3": ParagraphStyle(
            "rz_h3", parent=styles["Heading3"],
            fontSize=8, spaceBefore=6, spaceAfter=2,
            textColor=colors.HexColor("#34495e"),
        ),
        "meta": ParagraphStyle(
            "rz_meta", parent=styles["Normal"],
            fontSize=7, textColor=colors.gray, spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "rz_small", parent=styles["Normal"],
            fontSize=7, textColor=colors.HexColor("#555555"),
        ),
    }


def generate_pdf_report(
    img, mask, zones, row_length_cm, windows, all_feats, params,
    img_raw=None, debug_data=None,
):
    """
    Build and return the PDF report as raw bytes.

    Parameters
    ----------
    img           : np.ndarray  preprocessed RGB image
    mask          : np.ndarray  rock mask (same HxW)
    zones         : list of zone dicts
    row_length_cm : float
    windows       : list of (x1, x2) tuples
    all_feats     : list of feature dicts (one per window)
    params        : dict  sidebar parameter values
    img_raw       : np.ndarray or None  original image (for fracture_origin)
    debug_data    : dict or None  — if present and debug_data["enabled"] is True,
                    debug views are appended. Expected keys:
                      "enabled", "raw_labels", "smoothed",
                      "detect_wood", "wood_src", "wood_blocks",
                      "wood_brightness_min", "wood_uniformity_max", "w"
    """
    _, w_img = img.shape[:2]
    buf_out = io.BytesIO()
    ST = _make_styles()
    story = []

    # ── Page 1: summary ───────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    h_img, w_img = img.shape[:2]

    story.append(Paragraph("Rock Core Row Zonifier — Informe de análisis", ST["title"]))
    story.append(Paragraph(
        f"Longitud: <b>{row_length_cm:.0f} cm</b> &nbsp;·&nbsp; "
        f"Ventanas: <b>{len(windows)}</b> &nbsp;·&nbsp; "
        f"Zonas: <b>{len(zones)}</b> &nbsp;·&nbsp; "
        f"Imagen: {w_img}×{h_img} px &nbsp;·&nbsp; "
        f"Generado: {now}",
        ST["meta"],
    ))
    story.append(HRFlowable(
        width="100%", thickness=0.5, color=colors.lightgrey,
        spaceBefore=2, spaceAfter=6,
    ))

    # Row image — use actual PNG aspect ratio to avoid stretching
    story.append(Paragraph("Imagen original (preprocesada)", ST["h2"]))
    fig_row = render_row_with_ruler(img, w_img, row_length_cm)
    row_buf, row_aspect = _fig_to_buf(fig_row)
    plt.close(fig_row)
    story.append(_rl_image(row_buf, row_aspect))   # always full USABLE_W
    story.append(Spacer(1, 4))

    # Zone bar — same width as row image so they align for comparison
    story.append(Paragraph("Clasificación de zonas", ST["h2"]))
    fig_bar = render_bar_with_ruler(w_img, zones, row_length_cm, bar_height=48)
    bar_buf, bar_aspect = _fig_to_buf(fig_bar)
    plt.close(fig_bar)
    story.append(_rl_image(bar_buf, bar_aspect))   # always full USABLE_W
    story.append(Spacer(1, 3))

    # Legend
    legend_tbl = Table(
        [["🟢 Intacto", "🟡 Fracturado", "🔴 Rubble", "🟫 Madera"]],
        colWidths=[USABLE_W / 4] * 4,
    )
    legend_tbl.setStyle(TableStyle([
        ("ALIGN",    (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(legend_tbl)
    story.append(Spacer(1, 8))

    # Zone summary table
    story.append(Paragraph("Resumen de zonas", ST["h2"]))

    col_ws = [USABLE_W * f for f in [0.13, 0.12, 0.12, 0.12, 0.38, 0.13]]
    table_data = [["Zona", "Inicio", "Fin", "Longitud", "Origen fractura", "Confianza"]]

    for z in zones:
        lbl     = z["label"]
        lbl_esp = _LABEL_ESP[lbl]
        cs, ce  = z.get("cm_start"), z.get("cm_end")
        start_s  = f"{cs:.1f} cm" if cs is not None else f"{z['pct_start']*100:.1f}%"
        end_s    = f"{ce:.1f} cm" if ce is not None else f"{z['pct_end']*100:.1f}%"
        length_s = (f"{ce - cs:.1f} cm" if (cs is not None and ce is not None)
                    else f"{(z['pct_end'] - z['pct_start']) * 100:.1f}%")

        if lbl in ("fractured", "rubble") and img_raw is not None:
            p_nat, p_mec, conf, det = fracture_origin(img_raw, mask, z)
            nat_pct  = int(round(p_nat * 100))
            mec_pct  = int(round(p_mec * 100))
            dominant = "Natural" if p_nat >= p_mec else "Mecánica"
            reason   = (f"óxido {det['oxid_pct']:.1f}%"
                        if det["oxid_pct"] > 2
                        else f"brillo {det['mean_v']:.0f}/255")
            origin_s = f"{dominant} ({nat_pct}%N/{mec_pct}%M) · {reason}"
            conf_s   = conf
        else:
            origin_s = "—"
            conf_s   = "—"

        table_data.append([lbl_esp, start_s, end_s, length_s, origin_s, conf_s])

    tbl = Table(table_data, colWidths=col_ws, repeatRows=1)
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row_i, z in enumerate(zones, start=1):
        bg = _LABEL_BG.get(_LABEL_ESP[z["label"]])
        if bg:
            style_cmds.append(("BACKGROUND", (0, row_i), (0, row_i), bg))
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)

    # Parameters footer
    story.append(Spacer(1, 8))
    story.append(HRFlowable(
        width="100%", thickness=0.3, color=colors.lightgrey,
        spaceBefore=2, spaceAfter=3,
    ))
    story.append(Paragraph(
        f"<b>Parámetros:</b> "
        f"Ventana {params.get('window_width_pct', '?')}% · "
        f"Stride {params.get('stride_pct', '?')}% · "
        f"Max crack intacto {params.get('intact_crack_max', 0):.2f} · "
        f"Min crack rubble {params.get('rubble_crack_min', 0):.2f} · "
        f"Sobel σ={params.get('sobel_sigma', 0):.1f} · "
        f"Peso Sobel {params.get('sobel_weight', 0):.2f} · "
        f"Min intacto {params.get('min_intact_cm', 0):.0f} cm · "
        f"Suavizado {params.get('smoothing_win', '?')} · "
        f"Min zona {params.get('min_zone_wins', '?')} ventanas",
        ST["small"],
    ))

    # ── Debug pages ───────────────────────────────────────────────────────────
    if debug_data and debug_data.get("enabled"):
        raw_labels = debug_data["raw_labels"]
        smoothed   = debug_data["smoothed"]

        story.append(PageBreak())
        story.append(Paragraph("Debug — Vistas paso a paso", ST["title"]))
        story.append(HRFlowable(
            width="100%", thickness=0.5, color=colors.lightgrey,
            spaceBefore=2, spaceAfter=6,
        ))

        debug_figs = render_debug_views(
            img, mask, windows, all_feats, raw_labels, smoothed, zones
        )
        DEBUG_TITLES = [
            ("raw_row",          "1 · Imagen original"),
            ("rock_mask",        "2 · Máscara de roca"),
            ("sliding_windows",  "3 · Ventanas deslizantes"),
            ("feature_profiles", "4 · Perfiles de características"),
            ("raw_labels",       "5 · Etiquetas brutas (antes de suavizado)"),
            ("smoothed_labels",  "6 · Etiquetas suavizadas"),
            ("final_zones",      "7 · Zonas finales"),
        ]

        # Two figures per page: pair them up
        pairs = list(zip(DEBUG_TITLES[::2], DEBUG_TITLES[1::2]))
        # handle odd count
        if len(DEBUG_TITLES) % 2:
            pairs.append((DEBUG_TITLES[-1], None))

        MAX_H_EACH = (USABLE_H - 40) / 2  # half page height minus headings

        for (key1, title1), pair2 in pairs:
            story.append(Paragraph(title1, ST["h3"]))
            fig1 = debug_figs[key1]
            buf1, asp1 = _fig_to_buf(fig1)
            plt.close(fig1)
            story.append(_rl_image(buf1, asp1, max_h=MAX_H_EACH))
            story.append(Spacer(1, 6))

            if pair2:
                key2, title2 = pair2
                story.append(Paragraph(title2, ST["h3"]))
                fig2 = debug_figs[key2]
                buf2, asp2 = _fig_to_buf(fig2)
                plt.close(fig2)
                story.append(_rl_image(buf2, asp2, max_h=MAX_H_EACH))

            story.append(PageBreak())

        # ── Wood diagnostic (if applicable) ──────────────────────────────────
        if debug_data.get("detect_wood") and debug_data.get("wood_src") is not None:
            story.append(Paragraph("8 · Diagnóstico bloques de madera", ST["h3"]))

            wood_src          = debug_data["wood_src"]
            wood_blocks       = debug_data.get("wood_blocks", [])
            brightness_min    = debug_data.get("wood_brightness_min", 175)
            uniformity_max    = debug_data.get("wood_uniformity_max", 90)
            w_px              = debug_data.get("w", wood_src.shape[1])

            from core.wood import wood_column_profiles
            profiles = wood_column_profiles(wood_src, mask)
            xs = np.arange(w_px)

            fig_wd, ax_wd = plt.subplots(figsize=(14, 2.5))
            ax_wd.plot(xs, profiles["col_p90"],       label="p90",          color="gold",      lw=1.5)
            ax_wd.plot(xs, profiles["col_p10"],       label="p10",          color="sienna",    lw=1.2, ls="--")
            ax_wd.plot(xs, profiles["col_contrast"],  label="contraste",    color="tomato",    lw=1.5)
            ax_wd.plot(xs, profiles["bimodal"] * 255, label="bimodalidad×255", color="steelblue", lw=1.2, ls=":")
            ax_wd.axhline(brightness_min, color="gold",   ls="--", lw=0.8, alpha=0.6, label=f"umbral brillo ({brightness_min})")
            ax_wd.axhline(uniformity_max, color="tomato", ls="--", lw=0.8, alpha=0.6, label=f"umbral contraste ({uniformity_max})")
            for wb in wood_blocks:
                ax_wd.axvspan(wb["x_start"], wb["x_end"], alpha=0.25, color="brown")
            ax_wd.set_xlim(0, w_px)
            ax_wd.set_ylim(0, 270)
            ax_wd.set_xlabel("Posición (px)", fontsize=8)
            ax_wd.set_ylabel("Valor (0–255)", fontsize=8)
            ax_wd.legend(fontsize=7, ncol=3, loc="upper left")
            ax_wd.grid(True, alpha=0.2)
            fig_wd.tight_layout()

            buf_wd, asp_wd = _fig_to_buf(fig_wd)
            plt.close(fig_wd)
            story.append(_rl_image(buf_wd, asp_wd, max_h=MAX_H_EACH))

        # Remove trailing PageBreak if last item
        if story and isinstance(story[-1], PageBreak):
            story.pop()

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        buf_out, pagesize=PAGE,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=14 * mm, bottomMargin=12 * mm,
    )
    doc.build(story)
    buf_out.seek(0)
    return buf_out.read()
