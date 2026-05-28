"""
Rock Core Row Zonifier – Streamlit App
Zone-first pipeline: intact / fractured / rubble
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from core.preprocess import load_row_image, preprocess_row, resize_to_width
from core.mask       import build_rock_mask
from core.windows    import compute_sliding_windows
from core.features   import compute_all_window_features
from core.classify   import classify_windows, DEFAULT_THRESHOLDS
from core.smooth     import smooth_labels
from core.zones      import merge_labels_into_zones
from core.render     import (
    render_zone_bar, render_debug_views, fig_to_pil,
    render_row_with_ruler, render_bar_with_ruler,
)
from core.zones      import filter_short_intact, mark_wood_adjacent_as_mechanical
from core.wood       import detect_wood_blocks, apply_wood_to_zones, filter_wood_by_proximity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rock Core Zonifier",
    page_icon="🪨",
    layout="wide",
)

st.title("🪨 Rock Core Row Zonifier")
st.caption("Upload a single pre-cropped row image → get a green / yellow / red zone bar.")
st.info("🔖 v0.2 — 2026-04-26")

# ── Sidebar – parameters ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    row_length_input = st.number_input(
        "Row length (cm)",
        min_value=0.0, max_value=2000.0, value=0.0, step=1.0,
        help="Deja en 0 para usar el valor por defecto de 100 cm (1 m estándar).",
    )
    if row_length_input > 0:
        row_length_cm = float(row_length_input)
        _scale_note = None
    else:
        row_length_cm = 100.0
        _scale_note = "⚠️ Longitud no indicada — usando **100 cm (1 m)** como valor estándar."

    min_intact_cm = st.number_input(
        "Mínimo tramo intacto (cm)",
        min_value=1.0, max_value=50.0, value=10.0, step=1.0,
        help="Tramos intactos más cortos que este valor se reclasifican como Fractured. Estándar RQD = 10 cm.",
    )
    bypass_mech_rqd = st.checkbox(
        "🔧 Bypass roturas mecánicas para RQD",
        value=True,
        help="Las roturas de origen mecánico (drill) no cortan el tramo a efectos de medir si supera el mínimo. "
             "Se siguen mostrando en amarillo pero el tramo verde se mide incluyéndolas.",
    )

    st.divider()
    st.subheader("Windows")
    window_width_pct = st.slider("Window width (% of row)",  1, 20, 2, 1)
    stride_pct       = st.slider("Stride (% of row)",        1, 10, 1, 1)

    st.subheader("Smoothing")
    smoothing_win  = st.slider("Isolated-noise filter (# windows)", 1, 5, 1, 1,
                               help="Only kills lone outlier windows. Keep low to preserve narrow fractures.")
    min_zone_wins  = st.slider("Min zone size (# windows)",        1, 20, 2, 1)

    st.subheader("Crack thresholds  🔑")
    st.caption("Primary signal for solid-rock rows")
    intact_crack_max = st.slider(
        "Intact – max dark-column fraction",
        0.01, 0.20, 0.10, 0.01,
        help="Columns this much darker than the median are counted as cracks. "
             "Below this → intact.",
    )
    rubble_crack_min = st.slider(
        "Rubble – min dark-column fraction",
        0.10, 0.50, 0.25, 0.01,
        help="Above this → rubble.",
    )

    st.subheader("Mass-based thresholds")
    intact_cont_min = st.slider("Intact – min continuity score",  0.30, 0.90, 0.65, 0.05)
    intact_occ_min  = st.slider("Intact – min rock occupancy",    0.30, 0.90, 0.55, 0.05)
    rubble_frag_min = st.slider("Rubble – min fragmentation",     0.30, 0.90, 0.55, 0.05)

    st.subheader("Señal Sobel 〰️")
    st.caption("Detector de hairline cracks por gradiente")
    sobel_sigma = st.slider(
        "Umbral σ (sensibilidad)",
        1.0, 4.0, 1.3, 0.1,
        help="Nº de desviaciones estándar por encima de la media para considerar una columna como crack. "
             "Mayor → solo detecta gradientes extremos (menos falsos positivos en granito).",
    )
    sobel_weight = st.slider(
        "Peso Sobel vs. brillo",
        0.0, 1.0, 0.55, 0.05,
        help="0 = solo señal de brillo (original). 1 = solo Sobel. "
             "0.25 = Sobel aporta 25%, brillo 75%.",
    )

    st.divider()
    st.subheader("Imagen")
    do_resize = st.checkbox("Redimensionar imagen antes de procesar", value=False)
    resize_width = None
    if do_resize:
        resize_width = st.number_input(
            "Ancho objetivo (px)", min_value=300, max_value=4000, value=1200, step=100,
            help="Se mantiene el aspect ratio. Recomendado: 800–1500 px para imágenes de alta resolución.",
        )

    st.divider()
    st.subheader("🟫 Bloques de madera")
    detect_wood = st.checkbox("Detectar bloques de madera", value=True)
    if detect_wood:
        wood_brightness_min = st.slider(
            "Brillo mínimo (p90)", 140, 254, 175, 5,
            help="Percentil 90 de brillo por columna. Bajar si el CLAHE oscurece el bloque.",
        )
        wood_uniformity_max = st.slider(
            "Contraste mínimo tinta/papel (p90−p10)", 40, 200, 90, 10,
            help="Diferencia entre píxeles claros y oscuros dentro de la columna.",
        )
        wood_min_height = st.slider(
            "Cobertura vertical mínima (%)", 20, 80, 80, 5,
            help="El bloque debe aparecer en al menos este % de las filas.",
        ) / 100.0
        wood_bimodal_min = st.slider(
            "Bimodalidad mínima (%)", 20, 80, 60, 5,
            help="% de píxeles en los extremos (muy brillantes O muy oscuros). "
                 "Papel+tinta → alto. Roca → bajo. Bajar si no detecta.",
        ) / 100.0
    else:
        wood_brightness_min = 200
        wood_uniformity_max = 80
        wood_min_height     = 0.30
        wood_bimodal_min    = 0.35

    st.divider()
    debug_mode         = st.checkbox("🔍 Debug mode (step-by-step views)", value=True)
    show_fracture_exp  = st.checkbox("🔬 Señales experimentales de fractura", value=True)

# ── Input: staged row from Photo Cores OR file upload ────────────────────────
staged = st.session_state.get("staged_row")

if staged is not None:
    st.success(f"📦 Row staged from **{staged['source']}**")
    cols = st.columns([1, 1, 4])
    if cols[0].button("◀ Back to Photo Cores"):
        del st.session_state["staged_row"]
        st.switch_page("app.py")
    if cols[1].button("✕ Discard staged row"):
        del st.session_state["staged_row"]
        st.rerun()
    st.image(staged["img"], caption="Staged row preview", width="stretch")
    if not st.button("▶ Process this row", type="primary"):
        st.stop()
    img_raw_input = staged["img"]      # already an RGB numpy array
else:
    uploaded = st.file_uploader(
        "Upload a rock core row image (already cropped to a single row)",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )
    if not uploaded:
        st.info("Upload a row image, or stage one from the Photo Cores page.")
        st.stop()
    if not st.button("▶ Process", type="primary"):
        st.stop()
    img_raw_input = uploaded

with st.spinner("Analysing row…"):

    # 1 – Load & preprocess (accepts both numpy arrays and file objects)
    img_raw = (img_raw_input if isinstance(img_raw_input, np.ndarray)
               else load_row_image(img_raw_input))
    img     = preprocess_row(img_raw, apply_clahe=True, crop_margins=True)
    if do_resize and resize_width:
        img = resize_to_width(img, int(resize_width))
    h, w    = img.shape[:2]

    # 2 – Rock mask
    mask = build_rock_mask(img)

    # 3 – Sliding windows
    windows = compute_sliding_windows(
        w,
        window_width_ratio=window_width_pct / 100.0,
        stride_ratio=stride_pct / 100.0,
    )

    # 4 – Features
    all_feats = compute_all_window_features(
        img, mask, windows,
        sobel_sigma=sobel_sigma,
        sobel_weight=sobel_weight,
    )

    # 5 – Classify
    thresholds = {
        **DEFAULT_THRESHOLDS,
        "intact_crack_max":         intact_crack_max,
        "rubble_crack_min":         rubble_crack_min,
        "intact_continuity_min":    intact_cont_min,
        "intact_occupancy_min":     intact_occ_min,
        "rubble_fragmentation_min": rubble_frag_min,
    }
    raw_labels = classify_windows(all_feats, thresholds)

    # 6 – Smooth
    smoothed = smooth_labels(raw_labels, smoothing_win, min_zone_wins)

    # 7 – Zones
    zones = merge_labels_into_zones(smoothed, windows, w, row_length_cm)

    # 8 – Wood block detection (before enrichment so wood zones exist first)
    wood_blocks = []
    if detect_wood:
        import cv2 as _cv2
        wood_src = img_raw
        if wood_src.shape[1] != w:
            wood_src = _cv2.resize(
                wood_src, (w, img.shape[0]), interpolation=_cv2.INTER_AREA
            )
        wood_blocks = detect_wood_blocks(
            wood_src, mask,
            brightness_min=wood_brightness_min,
            uniformity_max=wood_uniformity_max,
            min_height_ratio=wood_min_height,
            bimodal_min=wood_bimodal_min,
        )
        wood_blocks = filter_wood_by_proximity(wood_blocks, w, row_length_cm, max_dist_cm=50.0)
        zones = apply_wood_to_zones(zones, wood_blocks, w, row_length_cm)

    # 9 – Enrich ALL fractured/rubble zones with fracture origin.
    #     Done AFTER wood detection so zone dicts are final (apply_wood_to_zones
    #     recreates dicts and would wipe any earlier enrichment).
    from core.fracture_signals import fracture_origin as _fracture_origin
    for _z in zones:
        if _z["label"] in ("fractured", "rubble"):
            _p_nat, _p_mec, _conf, _det = _fracture_origin(img_raw, mask, _z)
            _z["p_nat"]       = _p_nat
            _z["p_mec"]       = _p_mec
            _z["origin_conf"] = _conf
            _z["origin_det"]  = _det

    # 10 – Zones adjacent to wood blocks are always mechanical (drill-run boundary)
    if detect_wood:
        zones = mark_wood_adjacent_as_mechanical(zones)

    # 11 – Reclassify short intact zones (RQD logic, uses p_nat/p_mec from step 9)
    zones = filter_short_intact(zones, min_intact_cm, bypass_mechanical=bypass_mech_rqd)

# ── Results ───────────────────────────────────────────────────────────────────
if _scale_note:
    st.info(_scale_note)

st.success(
    f"Image: **{w} × {h} px**  |  Windows: **{len(windows)}**  |  Zones: **{len(zones)}**"
)

st.subheader("Original row")
fig_row = render_row_with_ruler(img, w, row_length_cm)
st.pyplot(fig_row, width="stretch")
plt.close(fig_row)

st.subheader("Zone classification")
fig_bar = render_bar_with_ruler(w, zones, row_length_cm, bar_height=48)
st.pyplot(fig_bar, width="stretch")
plt.close(fig_bar)

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown("🟢 **I** — Intact")
col2.markdown("🟡 **J** — Natural Joint")
col3.markdown("🟡 **MJ** — Mechanical Joint")
col4.markdown("🔴 **R** — Rubble")
col5.markdown("🟫 **W** — Wood block")

# ── Zone summary table ────────────────────────────────────────────────────────
st.subheader("Zone summary")

from core.fracture_signals import fracture_origin

EMOJI = {"intact": "🟢", "fractured": "🟡", "rubble": "🔴", "wood": "🟫"}
CONF_COLOR = {"alta": "#27ae60", "media": "#e67e22", "baja": "#95a5a6"}

for z in zones:
    lbl       = z["label"]
    width_pct = (z["pct_end"] - z["pct_start"]) * 100

    if row_length_cm:
        pos = f"{z['cm_start']:.1f} – {z['cm_end']:.1f} cm"
    else:
        pos = f"{z['pct_start']*100:.1f}% – {z['pct_end']*100:.1f}%"

    st.markdown(
        f"{EMOJI[lbl]} **{lbl.capitalize()}** &nbsp; {pos} &nbsp; "
        f"<span style='color:gray'>({width_pct:.1f}% of row)</span>",
        unsafe_allow_html=True,
    )

    if lbl in ("fractured", "rubble"):
        if "p_nat" in z:   # already computed during enrichment step
            p_nat, p_mec, conf, det = z["p_nat"], z["p_mec"], z["origin_conf"], z["origin_det"]
        else:              # zone created after enrichment (e.g. split by wood block)
            p_nat, p_mec, conf, det = fracture_origin(img_raw, mask, z)
        nat_pct    = int(round(p_nat * 100))
        mec_pct    = int(round(p_mec * 100))
        conf_color = CONF_COLOR[conf]
        dominant   = "🌿 Natural" if p_nat >= p_mec else "⚙️ Mecánica"
        # Explain which signals drove the result
        if det and det.get("forced") == "adjacent_to_wood":
            reason_str = "🟫 adyacente a bloque de madera → mecánica"
        else:
            reason_parts = []
            if det and det.get("oxid_pct", 0) > 2:
                reason_parts.append(f"óxido {det['oxid_pct']:.1f}% → natural")
            else:
                reason_parts.append("sin óxido → mecánica")
            if det and det.get("fresh_norm", 0) > 0.25:
                reason_parts.append(f"superficie brillante ({det['mean_v']:.0f}/255) → mecánica")
            elif det and det.get("fresh_norm", 0) < 0.1:
                reason_parts.append(f"superficie oscura ({det['mean_v']:.0f}/255)")
            reason_str = " · ".join(reason_parts)
        st.markdown(
            f"<div style='margin-left:28px; font-size:13px; color:#555'>"
            f"&nbsp;&nbsp;↳ {dominant} &nbsp;|&nbsp; "
            f"🌿 Natural <b>{nat_pct}%</b> &nbsp; "
            f"⚙️ Mecánica <b>{mec_pct}%</b> &nbsp; "
            f"<span style='color:{conf_color}'>confianza {conf}</span><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size:11px; color:#888'>{reason_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── PDF Export ────────────────────────────────────────────────────────────────
st.divider()
from core.report import generate_pdf_report

with st.spinner("Preparando informe PDF…"):
    _pdf_params = {
        "window_width_pct": window_width_pct,
        "stride_pct":       stride_pct,
        "intact_crack_max": intact_crack_max,
        "rubble_crack_min": rubble_crack_min,
        "intact_cont_min":  intact_cont_min,
        "intact_occ_min":   intact_occ_min,
        "rubble_frag_min":  rubble_frag_min,
        "sobel_sigma":      sobel_sigma,
        "sobel_weight":     sobel_weight,
        "smoothing_win":    smoothing_win,
        "min_zone_wins":    min_zone_wins,
        "min_intact_cm":    min_intact_cm,
    }
    _pdf_debug = {
        "enabled":            debug_mode,
        "raw_labels":         raw_labels,
        "smoothed":           smoothed,
        "detect_wood":        detect_wood,
        "wood_src":           wood_src if detect_wood else None,
        "wood_blocks":        wood_blocks,
        "wood_brightness_min": wood_brightness_min,
        "wood_uniformity_max": wood_uniformity_max,
        "w":                  w,
    }
    _pdf_bytes = generate_pdf_report(
        img, mask, zones, row_length_cm,
        windows, all_feats, _pdf_params,
        img_raw=img_raw,
        debug_data=_pdf_debug,
    )

st.download_button(
    label="📄 Descargar informe PDF",
    data=_pdf_bytes,
    file_name="rock_zonifier_report.pdf",
    mime="application/pdf",
)

# ── Fracture signal comparison (experimental) ─────────────────────────────────
if show_fracture_exp:
    st.divider()
    st.header("🔬 Señales de fractura — 4 enfoques independientes")
    st.caption(
        "Cada señal es independiente del pipeline principal. "
        "El fondo de cada gráfico muestra la clasificación actual (verde=intacto, amarillo=fracturado, rojo=rubble)."
    )

    from core.fracture_signals import (
        hough_line_signal,
        glcm_texture_signal,
        colour_oxidation_signal,
        morphology_crack_signal,
        sobel_gradient_signal,
        sobel_gradient_image,
    )

    with st.spinner("Calculando señales experimentales…"):
        sig_hough  = hough_line_signal(img, mask, windows)
        sig_glcm   = glcm_texture_signal(img, mask, windows)
        sig_colour = colour_oxidation_signal(img_raw, mask, windows)
        sig_morph  = morphology_crack_signal(img, mask, windows)
        sig_sobel  = sobel_gradient_signal(img, mask, windows)
        sobel_img  = sobel_gradient_image(img, mask)
        sig_crack  = np.array([f["crack_column_fraction"] for f in all_feats])
        sig_crack  = sig_crack / sig_crack.max() if sig_crack.max() > 0 else sig_crack

    xs_cm = np.array([(x1 + x2) / 2 / w * row_length_cm for x1, x2 in windows])

    ZONE_COLORS = {"intact": "#2ecc40", "fractured": "#ffaa00", "rubble": "#e74c3c", "wood": "#8b5a2b"}

    def _add_zone_background(ax, zones, row_length_cm):
        for z in zones:
            ax.axvspan(z["cm_start"], z["cm_end"],
                       color=ZONE_COLORS.get(z["label"], "gray"),
                       alpha=0.12, zorder=0)

    SIGNALS = [
        (sig_hough,  "1 · Líneas Hough",              "steelblue",   ),
        (sig_glcm,   "2 · Textura GLCM",               "darkorange",  ),
        (sig_colour, "3 · Oxidación (color)",          "saddlebrown", ),
        (sig_morph,  "4 · Forma de regiones oscuras",  "purple",      ),
        (sig_sobel,  "5 · Gradiente Sobel (hairline)", "crimson",     ),
    ]

    fig_exp, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig_exp.suptitle("Señales experimentales de fractura", fontsize=12)

    for ax, (sig, title, color) in zip(axes, SIGNALS):
        _add_zone_background(ax, zones, row_length_cm)
        ax.plot(xs_cm, sig, color=color, lw=1.5, label=title)
        ax.plot(xs_cm, sig_crack, color="gray", lw=0.8, ls=":", alpha=0.5, label="crack pipeline (ref.)")
        ax.set_ylabel("Señal [0–1]", fontsize=8)
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, row_length_cm)

    axes[-1].set_xlabel("Posición (cm)")
    fig_exp.tight_layout()
    st.pyplot(fig_exp, width="stretch")
    plt.close(fig_exp)

    st.caption(
        "**Interpretación**: si varias señales muestran picos en el mismo cm → alta confianza de fractura real. "
        "Señal 3 (oxidación) alta → fractura natural. "
        "Señales 1+4+5 altas pero señal 3 baja → posible fractura mecánica (roca fresca)."
    )

    # ── Visualización Sobel para fine-tuning ─────────────────────────────────
    st.subheader("🔬 Imagen de gradiente Sobel")
    st.caption(
        "Las zonas brillantes son bordes verticales fuertes — ahí el algoritmo 've' transiciones bruscas. "
        "Los hairline cracks aparecen como líneas brillantes aunque no sean oscuros en la imagen original."
    )

    _sobel_row_h = max(2.0, 2 * (14 * h / w) + 0.8)
    fig_sobel, axes_s = plt.subplots(
        2, 1, figsize=(14, _sobel_row_h),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    axes_s[0].imshow(img, aspect="auto")
    axes_s[0].set_title("Original (preprocesada)", fontsize=9)
    axes_s[0].axis("off")

    axes_s[1].imshow(sobel_img, cmap="hot", aspect="auto")
    axes_s[1].set_title("Gradiente Sobel X  (brillante = borde fuerte)", fontsize=9)
    axes_s[1].axis("off")

    # Ruler on Sobel image
    tick_cms  = np.arange(0, row_length_cm + 1, 10)
    tick_pxs  = (tick_cms / row_length_cm * w).astype(int)
    axes_s[1].set_xticks(tick_pxs)
    axes_s[1].set_xticklabels([f"{c:.0f}" for c in tick_cms], fontsize=7)
    axes_s[1].set_xlabel("Posición (cm)", fontsize=8)
    axes_s[1].tick_params(axis="x", length=3)

    fig_sobel.tight_layout(pad=0.5)
    st.pyplot(fig_sobel, width="stretch")
    plt.close(fig_sobel)

# ── Debug views ───────────────────────────────────────────────────────────────
if debug_mode:
    st.divider()
    st.header("🔍 Debug views")

    debug_figs = render_debug_views(
        img, mask, windows, all_feats, raw_labels, smoothed, zones
    )

    DEBUG_TITLES = {
        "raw_row":          "1 · Raw Row",
        "rock_mask":        "2 · Rock Mask",
        "sliding_windows":  "3 · Sliding Windows",
        "feature_profiles": "4 · Feature Profiles",
        "raw_labels":       "5 · Raw Labels (before smoothing)",
        "smoothed_labels":  "6 · Smoothed Labels",
        "final_zones":      "7 · Final Zones",
    }

    for key, title in DEBUG_TITLES.items():
        st.subheader(title)
        fig = debug_figs[key]
        st.image(fig_to_pil(fig), width="stretch")
        plt.close(fig)

    # ── Wood block diagnostic ─────────────────────────────────────────────────
    if detect_wood:
        st.subheader("8 · Wood block diagnostic")
        st.caption("Perfiles por columna usados en la detección. "
                   "El bloque de madera debería verse como un pico en p90 y contraste.")
        from core.wood import wood_column_profiles
        profiles = wood_column_profiles(wood_src, mask)
        xs = np.arange(w)

        fig_wd, ax_wd = plt.subplots(figsize=(12, 3))
        ax_wd.plot(xs, profiles["col_p90"],      label="p90 (fondo papel)",  color="gold",      lw=1.5)
        ax_wd.plot(xs, profiles["col_p10"],      label="p10 (tinta/sombra)", color="sienna",    lw=1.2, ls="--")
        ax_wd.plot(xs, profiles["col_contrast"], label="contraste p90−p10",  color="tomato",    lw=1.5)
        ax_wd.plot(xs, profiles["bimodal"] * 255, label="bimodalidad ×255",  color="steelblue", lw=1.2, ls=":")
        ax_wd.axhline(wood_brightness_min, color="gold",    ls="--", lw=0.8, alpha=0.6, label=f"umbral brillo ({wood_brightness_min})")
        ax_wd.axhline(wood_uniformity_max, color="tomato",  ls="--", lw=0.8, alpha=0.6, label=f"umbral contraste ({wood_uniformity_max})")
        for wb in wood_blocks:
            ax_wd.axvspan(wb["x_start"], wb["x_end"], alpha=0.25, color="brown", label="bloque detectado")
        ax_wd.set_xlim(0, w)
        ax_wd.set_ylim(0, 270)
        ax_wd.set_xlabel("Posición (px)")
        ax_wd.set_ylabel("Valor (0–255)")
        ax_wd.legend(fontsize=7, ncol=3, loc="upper left")
        ax_wd.grid(True, alpha=0.2)
        st.image(fig_to_pil(fig_wd), width="stretch")
        plt.close(fig_wd)
