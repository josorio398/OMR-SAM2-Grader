# 🎓 OMR-SAM2-Grader  
**Calificador OMR con Segmentación Semántica (Grounding DINO + SAM-2) para exámenes de selección múltiple**

OMR-SAM2-Grader es un sistema de calificación automática de exámenes tipo **selección múltiple** que combina:
- **Detección *zero-shot*** (sin entrenamiento) con **Grounding DINO** para localizar burbujas de respuesta a partir de un *prompt*.
- **Segmentación semántica de alta precisión** con **SAM-2** para recortar exactamente la región de cada burbuja (más robusto que usar solo cajas).
- **Decisión de marcado por estadística de intensidad** dentro de cada máscara (A/B/C/D), reduciendo falsos positivos por sombras o ruido.
- **OCR (RapidOCR)** para extraer metadatos del encabezado (curso, documento, cuadernillo).

> En otras palabras: este proyecto no depende de plantillas rígidas con marcas antiguas; usa modelos modernos para **detectar + segmentar** los objetivos en la hoja y luego inferir la respuesta marcada.

---

## ✨ Características

- **Procesamiento masivo desde PDF** (PyMuPDF @ 300 DPI).
- **Detección de burbujas** con Grounding DINO (*zero-shot* + filtrado estructural).
- **Segmentación precisa** con SAM-2 usando `bboxes` (máscaras por burbuja).
- **Clasificación por intensidad** (A/B/C/D):
  - `Anulada` (más de una opción marcada)
  - `Sin respuesta` (diferencia insuficiente)
- **OCR del encabezado** (RapidOCR):
  - Curso
  - Número de documento
  - Número de cuadernillo
- **Exportación automática** a Excel (`Resultados_Masivos_Salon.xlsx`).
- **Debug visual completo**:
  - `debug/input/` → imágenes antes del procesamiento (por página)
  - `debug/output/` → overlays con máscaras/segmentación (por página)
  - ZIP descargable con todas las evidencias

---

## 🧠 Modelos utilizados

- **Grounding DINO**: `IDEA-Research/grounding-dino-base` (Transformers)
- **SAM-2**: `sam2_b.pt` (Ultralytics)
- **OCR**: RapidOCR (`rapidocr-onnxruntime`)

---

## 🚀 Google Colab — Instalación y ejecución

> Copia y pega estas celdas en Colab, en orden.

### **Instalación**
```python
!git clone https://github.com/josorio398/OMR-SAM2-Grader.git
%cd OMR-SAM2-Grader
!pip install -q -r requirements.txt

import torch
print(f"✅ Entorno listo en: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

!python scripts/smoke_test.py

from omr_sam2_grader.colab_entry import run
run()
