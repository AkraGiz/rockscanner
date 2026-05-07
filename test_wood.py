"""
Diagnóstico de detección de bloques de madera.
Uso:
    python test_wood.py <ruta_imagen>

Genera test_wood_output.png con los perfiles de brillo columna a columna.
Imprime en consola los valores máximos y los candidatos detectados.
"""

import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.preprocess import load_row_image, preprocess_row
from core.mask       import build_rock_mask


def run(image_path):
    # ── Carga ──────────────────────────────────────────────────────────────────
    print(f"\n📂 Cargando: {image_path}")
    img_raw = load_row_image(image_path)
    img     = preprocess_row(img_raw, apply_clahe=True, crop_margins=True)
    mask    = build_rock_mask(img)
    h, w    = img.shape[:2]
    print(f"   Tamaño: {w}×{h} px")

    # ── Usar imagen sin CLAHE para madera ──────────────────────────────────────
    src = img_raw
    if src.shape[:2] != (h, w):
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Enmascarar fondo (bandeja)
    masked = gray.astype(float)
    masked[mask == 0] = np.nan

    # ── Perfiles por columna ───────────────────────────────────────────────────
    col_p90      = np.nanpercentile(masked, 90, axis=0)
    col_p10      = np.nanpercentile(masked, 10, axis=0)
    col_p90      = np.where(np.isnan(col_p90), 0.0, col_p90)
    col_p10      = np.where(np.isnan(col_p10), 0.0, col_p10)
    col_contrast = col_p90 - col_p10

    # Bimodalidad por columna
    bimodal = np.zeros(w)
    for x in range(w):
        col = gray[:, x].astype(float)
        valid = mask[:, x] > 0
        if valid.any():
            pix = col[valid]
            bimodal[x] = ((pix > 215) | (pix < 55)).mean()

    # Cobertura vertical (% de filas con al menos un píxel brillante)
    vert_cov = np.zeros(w)
    for x in range(w):
        col = gray[:, x]
        vert_cov[x] = (col >= 150).mean()   # umbral fijo para diagnóstico

    # ── Consola: top-20 columnas por p90 ──────────────────────────────────────
    print("\n🔍 Top-20 columnas por p90 (candidatas a madera):")
    print(f"   {'Col':>6}  {'p90':>6}  {'p10':>6}  {'Contraste':>10}  {'Bimodal':>8}  {'CobVert':>8}")
    top20 = np.argsort(col_p90)[::-1][:20]
    for x in sorted(top20):
        print(f"   {x:>6}  {col_p90[x]:>6.1f}  {col_p10[x]:>6.1f}  "
              f"{col_contrast[x]:>10.1f}  {bimodal[x]:>8.3f}  {vert_cov[x]:>8.3f}")

    # ── Resumen de rangos ──────────────────────────────────────────────────────
    print(f"\n📊 Rangos globales:")
    print(f"   p90      → min: {col_p90.min():.1f}  max: {col_p90.max():.1f}  "
          f"mediana: {np.median(col_p90):.1f}")
    print(f"   contraste→ min: {col_contrast.min():.1f}  max: {col_contrast.max():.1f}  "
          f"mediana: {np.median(col_contrast):.1f}")
    print(f"   bimodal  → min: {bimodal.min():.3f}  max: {bimodal.max():.3f}  "
          f"mediana: {np.median(bimodal):.3f}")
    print(f"   cob.vert.→ min: {vert_cov.min():.3f}  max: {vert_cov.max():.3f}  "
          f"mediana: {np.median(vert_cov):.3f}")

    # ── Sugerencia de umbrales ─────────────────────────────────────────────────
    # Zona de interés: columnas con p90 significativamente por encima de la mediana
    threshold_suggestion = np.percentile(col_p90, 90)
    wood_cols = col_p90 >= threshold_suggestion
    print(f"\n💡 Umbral sugerido de brillo (p90 del propio perfil): {threshold_suggestion:.1f}")
    groups = []
    in_g, start = False, 0
    for x, v in enumerate(wood_cols):
        if v and not in_g:
            start, in_g = x, True
        elif not v and in_g:
            groups.append((start, x))
            in_g = False
    if in_g:
        groups.append((start, w))
    print(f"   Grupos candidatos con ese umbral:")
    for g in groups:
        pct_s = g[0] / w * 100
        pct_e = g[1] / w * 100
        print(f"     px {g[0]}–{g[1]}  ({pct_s:.1f}%–{pct_e:.1f}%,  ancho {g[1]-g[0]} px)")

    # ── Gráfico ────────────────────────────────────────────────────────────────
    xs = np.arange(w)
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"Diagnóstico de madera — {image_path}", fontsize=11)

    axes[0].plot(xs, col_p90, color="gold", lw=1.2, label="p90")
    axes[0].axhline(threshold_suggestion, color="gold", ls="--", lw=0.8, alpha=0.7,
                    label=f"umbral sugerido ({threshold_suggestion:.0f})")
    axes[0].set_ylabel("p90 brillo")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.2)

    axes[1].plot(xs, col_contrast, color="tomato", lw=1.2, label="contraste (p90−p10)")
    axes[1].set_ylabel("Contraste")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.2)

    axes[2].plot(xs, bimodal, color="steelblue", lw=1.2, label="bimodalidad")
    axes[2].set_ylabel("Bimodal")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.2)

    axes[3].plot(xs, vert_cov, color="mediumseagreen", lw=1.2, label="cobertura vertical")
    axes[3].set_ylabel("Cob. vertical")
    axes[3].set_xlabel("Posición (px)")
    axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.2)

    # Marcar grupos candidatos
    for g in groups:
        for ax in axes:
            ax.axvspan(g[0], g[1], alpha=0.15, color="brown")

    # Mostrar imagen en miniatura arriba
    fig2, ax2 = plt.subplots(figsize=(16, 1.5))
    ax2.imshow(src)
    ax2.set_title("Imagen original (sin CLAHE)", fontsize=9)
    ax2.axis("off")

    out1 = "test_wood_profiles.png"
    out2 = "test_wood_image.png"
    fig.savefig(out1, bbox_inches="tight", dpi=100)
    fig2.savefig(out2, bbox_inches="tight", dpi=100)
    plt.close("all")
    print(f"\n✅ Guardado: {out1}")
    print(f"✅ Guardado: {out2}")
    print("\nAbre esos archivos y busca visualmente dónde está el pico de p90.")
    print("Ese valor máximo de p90 será tu umbral de brillo óptimo.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_wood.py <ruta_imagen>")
        sys.exit(1)
    run(sys.argv[1])
