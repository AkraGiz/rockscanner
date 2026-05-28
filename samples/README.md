# Photo Cores — Sample Catalog

Esta carpeta agrupa fotografías reales de cajas de testigos para calibrar
y testear los algoritmos de detección automática y editor manual.

## Taxonomía

Cada subcarpeta agrupa cajas con características visuales / geométricas comunes:

| Carpeta | Descripción | Status MVP |
|---|---|---|
| `wood_horizontal/`     | Caja de madera con listones horizontales, rocas secas — **caso canónico** | ✅ Auto |
| `wood_vertical/`       | Caja de madera con canales en vertical (testigos en vertical) | ⚠️ Detectar orientación |
| `plastic_black/`       | Caja de plástico negro (roca cualquier color) | ⚠️ Editor manual |
| `metal_tubes/`         | Tubos individuales de aluminio / chapa galvanizada | ⚠️ Otra rama de detección |
| `cardboard/`           | Filas separadas por cartón ondulado (material blando) | 🚫 Fuera de MVP |
| `tricono_cuttings/`    | Cuttings de perforación tricónica (no son testigos cilíndricos) | 🚫 Detectar y rechazar |
| `soil_regolith/`       | Suelo / regolito / arcilla — no aplica RQD | 🚫 Detectar y rechazar |
| `wet/`                 | Rocas mojadas (de cualquier tipo de caja) — señales distintas | ⚠️ Toggle manual |
| `annotations_geological/` | Cajas con anotaciones rojas (buzamientos), texto mineral, bolsas blancas de assay | ⚠️ Filtrar como ruido |

## Cómo añadir muestras

1. Renombra las fotos siguiendo este patrón: `<proyecto>_<hole>_<box>_<descripción>.jpg`
   - Ejemplo: `galore_GCT-21-001_BX229_wet.jpg`
   - Si no sabes el proyecto/hole, usa cualquier identificador único: `sample_001.jpg`

2. Cópialas a la subcarpeta que mejor describa el caso.

3. Si una foto cubre múltiples casos (p. ej. wood box pero con rocas mojadas),
   guárdala en `wet/` y deja una nota corta en `notes.md` de esa carpeta.

## Casos de interés especial para validación

- `wood_horizontal/` debería contener al menos:
  - Una con header card profesional (Galore Creek style)
  - Una sin ningún tipo de cabecera
  - Una con maderitas estándar
  - Una con pegatinas en los bordes en vez de maderitas
  - Una con rocas claras y otra con rocas oscuras

- `metal_tubes/` debería contener:
  - Una con rocas oxidadas (naranja)
  - Una con rocas grises
  - Una con header card "DDH-XXX" estilo Hipógeno

- `plastic_black/` debería ser el peor caso posible: caja oscura, roca oscura.

## Cómo se usan estas muestras

- **Pruebas manuales** durante el desarrollo del módulo `core/corebox.py`
- **Test informal de regresión**: si tocamos detección y se rompe en una muestra
  que funcionaba, sabemos que hemos roto algo
- **Documentación viva** del muestrario que el sistema tiene que soportar

No es necesario subir todas las muestras al repo si pesan demasiado — un
puñado representativo por carpeta es suficiente.

## ⚠️ Las imágenes NO se versionan en git

Las imágenes de las subcarpetas están en `.gitignore` (extensiones `.jpg`,
`.png`, etc están globalmente excluidas).  La colección real puede llegar
fácilmente a varios GB, no caben en GitHub.

Solo se versionan:
- Este README
- Los archivos `.gitkeep` que mantienen la estructura de carpetas
- Cualquier `notes.md` que pongas dentro de cada subcarpeta

Si quieres compartir muestras con el equipo, usa OneDrive / Drive / S3 / etc.
