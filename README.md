# 🎓 OMR SAM-2 Grader

Sistema inteligente de calificación de exámenes de selección múltiple basado en IA de última generación.  
Utiliza **Segment Anything Model 2 (SAM-2)** para segmentación precisa de burbujas y **Grounding DINO** para detección *zero-shot* (sin entrenamiento).

✅ Diseñado para flujo **masivo desde PDF** (1 página = 1 estudiante).  
✅ Genera **tabla consolidada** + **descarga automática** en Excel.  
✅ Incluye evidencias visuales en carpeta `debug/` (**input** y **output**).

---

## ✨ Características

- **Procesamiento masivo desde PDF** (PyMuPDF @ 300 DPI).
- **Detección de burbujas** con Grounding DINO (prompt + filtrado estructural).
- **Segmentación precisa** con SAM-2 usando `bboxes`.
- **Decisión robusta de respuesta** por intensidad:
  - A/B/C/D
  - `Anulada` (más de una marcada)
  - `Sin respuesta` (diferencia insuficiente)
- **OCR del encabezado** (RapidOCR) para:
  - Curso
  - Número de documento
  - Número de cuadernillo
- **Exportación automática a Excel** (`Resultados_Masivos_Salon.xlsx`).
- **Debug visual**:
  - `debug/input/` → imagen por página antes del procesamiento (entrada)
  - `debug/output/` → overlay con máscaras detectadas (salida)

---

## 🚀 Uso rápido en Google Colab (recomendado)

### Opción A — Notebook listo (mínimo código)
1. Abre el notebook:
   - `notebooks/OMR_SAM2_Grader_Colab.ipynb`
2. Ejecuta las celdas en orden.
3. Sube tu PDF cuando se solicite.
4. Se descargan automáticamente:
   - `outputs/Resultados_Masivos_Salon.xlsx`
   - `outputs/Imagenes_Diagnostico.zip` (incluye `debug/input` y `debug/output`)

### Opción B — 3 pasos (manual en Colab)
```python
!git clone https://github.com/josorio398/OMR-SAM2-Grader.git
%cd OMR-SAM2-Grader
!pip install -q -r requirements.txt
