# Photo Cores — Diseño técnico

## Propósito

Extender Rock Core Zonifier para aceptar **fotografías completas de cajas
de testigo** (Photo Cores) y producir el análisis de fracturas que la app
ya hace, pero para cada fila de la caja en orden de lectura
(izquierda-derecha, arriba-abajo).

A largo plazo, varias cajas se concatenan virtualmente como un único
sondaje continuo.

## Estado actual del sistema

El pipeline existente (`app.py`) procesa **una sola fila ya recortada**:
- Toma una imagen de un strip de roca con un mask binario
- Aplica ventanas deslizantes y clasifica cada ventana
- Produce una barra de zonas (intacto / J / MJ / rubble / madera)
- Calcula origen natural vs mecánica y RQD con bypass de mecánicas

Esto seguirá funcionando como **modo "single row"** independientemente del
flujo nuevo. El nuevo flujo de Photo Cores **produce el strip de entrada**
para el pipeline existente, no lo sustituye.

## Variabilidad observada en muestras reales

Con ~30 fotos reales catalogadas, las dimensiones de variación son:

| Eje | Casos |
|---|---|
| Material caja | madera clara · madera oscura · plástico negro · aluminio · cartón |
| Geometría | horizontal · vertical · tubos independientes |
| Material muestra | cilindros frescos · rubble · cuttings · suelo · arcilla |
| Estado | seca · mojada |
| Marcadores | maderitas · pegatinas en bordes · texto sobre listón · etiquetas assay |
| Anotaciones | sin marcas · líneas rojas estructurales · texto mineral · bolsas blancas |
| Iluminación | sol directo · interior · sombra · contraluz |
| Header card | sin · cabecera grande · Galore-style con calibración de color |

**Conclusión clave**: la auto-detección al 100 % automática NO es realista.
La estrategia es **auto-detección como propuesta + editor manual como
fuente de verdad**.

## Estrategia en 3 fases

### Fase 1 — MVP funcional (este sprint)

Cubre solo el caso canónico: **caja de madera con listones horizontales,
rocas secas**. Todos los demás casos van directamente al editor manual.

- Detección automática del **contorno de la caja**
- Detección de los **listones horizontales** que separan los canales
- **Extracción de cada fila** como strip con su mask
- UI multipage de Streamlit con upload de N fotos
- Editor manual (drawable canvas) para corregir contorno y listones
- Cada fila extraída pasa al pipeline existente (single-row analyzer)

Casos fuera de fase 1:
- Vertical orientation
- Black plastic / metal tubes (van por la rama "editor manual" desde el inicio)
- OCR de header cards
- Wet rocks (toggle manual)
- Tricono / regolito (detección + rechazo)

### Fase 2 — Expansión de auto-detección

- Detección de orientación (horizontal vs vertical)
- Rama de detección específica para tubos metálicos
- OCR del header card (HOLE_ID, METERS_FROM/TO, BOX#)
- Detección de regla impresa al pie → calibración cm/px automática
- Toggle "roca mojada" que ajusta umbrales
- Detección de material no apto (cuttings, regolito) y aviso

### Fase 3 — Linealización del sondaje

- Catálogo de cajas como **proyecto** persistente
- Concatenación virtual de cajas en un único sondaje continuo
- RQD agregado del sondaje completo
- Export del informe PDF a nivel de proyecto

## Arquitectura

### Módulos nuevos

```
core/
  corebox.py           ← detección automática (contorno, listones, extracción)

pages/
  1_📦_Photo_Cores.py  ← UI multipage de Streamlit
```

### API pública de `core/corebox.py`

```python
def detect_box_outline(img: np.ndarray) -> np.ndarray:
    """
    Devuelve el polígono (4 vértices) o bounding box del contorno de la caja.
    Heurística: detección por bajo contraste con el fondo, edge detection,
    contorno mayor.
    """

def detect_horizontal_dividers(img: np.ndarray, box: np.ndarray) -> list[int]:
    """
    Devuelve las coordenadas Y de los listones que separan los canales.
    Heurística: proyección horizontal de intensidad dentro del box, picos
    periódicos. Aplica filtro de paralelismo y equiespaciado.
    """

def extract_rows(img: np.ndarray, box: np.ndarray, dividers: list[int]) -> list[Row]:
    """
    Recorta cada canal entre listones consecutivos.
    Devuelve list[Row] en orden de lectura (arriba a abajo) con la imagen
    de la fila, su mask de roca, y su bbox dentro de la foto original.
    """

@dataclass
class Row:
    img: np.ndarray
    mask: np.ndarray | None    # se construye después con build_rock_mask
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) en coords de la foto
    index: int                  # 0-based, orden de lectura
```

### Persistencia en sesión

El estado de cada Photo Core (contorno, listones, ediciones manuales,
resultados de análisis por fila) se guarda en `st.session_state` keyed
por hash de imagen. Esto permite:
- Subir varias fotos, editar una, volver a otra sin perder el trabajo
- Re-analizar sólo las filas modificadas
- Persistir las correcciones manuales entre recargas de la página

### Flujo de usuario (fase 1)

```
1. Usuario va a la página "📦 Photo Cores"
2. Sube N fotos de cajas
3. Para cada foto:
   a. Auto-detección de contorno + listones (rápida)
   b. Se muestra overlay sobre la foto original
   c. Botón "Editar contorno"     → abre drawable canvas, modo polígono
   d. Botón "Editar listones"     → abre drawable canvas, modo línea horizontal
   e. Botón "Analizar fila X"     → recorta y pasa al pipeline single-row
4. Resultados de cada fila se muestran inline + se pueden exportar a PDF
```

## Decisiones clave

### 1. Detección conservadora (alta precisión, baja recall)

Mejor que la auto-detección marque DE MENOS y el usuario añada, que no que
marque DE MÁS y el usuario tenga que borrar. Es 5× más rápido añadir 3
listones que borrar 7 falsos positivos.

### 2. Editor manual como ciudadano de primera clase

No es un fallback. Es una herramienta de trabajo principal. Se invierte
en su UX (atajos de teclado, deshacer, snap a guías auto) desde el principio.

### 3. La auto-detección y la edición son la misma representación

El estado interno de un Photo Core es siempre el mismo dict, viva como auto
o como manual:

```python
{
    "outline": [(x,y), (x,y), (x,y), (x,y)],   # 4 vértices
    "dividers": [y1, y2, y3, y4, y5],          # Y coords
    "source": "auto" | "manual",
    "edits": [...]                              # historial para deshacer
}
```

### 4. Sin OCR en fase 1

El usuario rellena manualmente HOLE_ID, METERS_FROM, METERS_TO, BOX# en
campos de la UI. OCR se añade en fase 2 como un "pre-llenado".

### 5. Cuttings / regolito en fase 1 no se detectan

Sería deseable que el sistema reconozca "esto no es roca, no aplica RQD"
y avise. En fase 1 confiamos en que el usuario no suba ese tipo de imagen.

## Casos manejados y no manejados — fase 1

### ✅ Manejados automáticamente

- Caja madera horizontal, rocas secas, iluminación normal
- 3 a 8 canales por caja
- Maderitas / pegatinas como marcadores dentro de canal (ya lo hace el pipeline existente)

### ⚠️ Manejados vía editor manual

- Caja vertical (usuario rota o redibuja)
- Caja plástico negro / aluminio
- Rocas mojadas (visualmente diferentes pero el editor sigue funcionando)
- Header cards / etiquetas que tape parte de la caja (usuario lo recorta del polígono)

### 🚫 Fuera de fase 1

- Tricono cuttings / suelo / regolito (no es nuestro target geológico)
- Detección de orientación automática
- OCR
- Concatenación cross-box

## Riesgos identificados

1. **Detección de listones falla en cajas con sombras fuertes** — la proyección
   horizontal asume iluminación uniforme. Mitigación: ecualización local de
   histograma antes de proyectar.

2. **streamlit-drawable-canvas tiene latencia con imágenes grandes** — fotos
   reales son ~4000 px. Mitigación: trabajar sobre versión reducida en la UI,
   re-escalar las coordenadas a la imagen original al guardar.

3. **Persistencia de session_state se pierde al cerrar el navegador** —
   ediciones se evaporan. Mitigación a futuro: persistir a disco (sqlite)
   en fase 3 cuando introduzcamos "proyectos".

## Open questions

- **¿Cuánto pesa una foto típica?** Si superan 10MB en serio, hay que recortar
  agresivamente antes de meterlas en Streamlit.
- **¿La longitud de cada fila puede asumirse 1 m?** Algunas cajas son de
  1 m por fila estándar pero otras pueden ser distintas. Por defecto 1 m
  con override por fila.
- **¿Las fotos vienen siempre top-down razonable?** Si hay perspectiva fuerte
  hay que añadir corrección por homografía.
