# 🎓 OMR SAM-2 Grader

Sistema inteligente de calificación de exámenes de selección múltiple basado en IA de última generación.  
Utiliza **Segment Anything Model 2 (SAM-2)** para segmentación precisa de burbujas y **Grounding DINO** para detección *zero-shot* (sin entrenamiento).

> ✅ Diseñado para flujo **masivo desde PDF** (1 página = 1 estudiante).  
> ✅ Genera tabla consolidada + descarga automática en Excel.  
> ✅ Incluye evidencias visuales en carpeta `debug/`.

---

## ✨ Características

- **Procesamiento masivo desde PDF** (PyMuPDF @ 300 DPI).
- **Detección de burbujas** con Grounding DINO (prompt + filtrado estructural).
- **Segmentación precisa** con SAM-2 usando `bboxes`.
- **Decisión robusta de respuesta** por intensidad (A/B/C/D, Anulada, Sin respuesta).
- **OCR del encabezado** (RapidOCR) para:
  - Curso
  - Número de documento
  - Número de cuadernillo
- **Exportación automática a Excel** (`Resultados_Masivos_Salon.xlsx`).
- **Debug visual**:
  - `debug/input/` → imagen preprocesada por página
  - `debug/output/` → overlay con máscaras detectadas

---

## 🚀 Uso rápido en Google Colab (recomendado)

### Opción A — Notebook listo (mínimo código)
1. Abre: `notebooks/OMR_SAM2_Grader_Colab.ipynb`
2. Ejecuta las celdas en orden.
3. Sube tu PDF cuando te lo solicite.
4. Descarga automáticamente:
   - `Resultados_Masivos_Salon.xlsx`
   - `Imagenes_Diagnostico.zip`

### Opción B — 3 líneas en una celda
```python
!git clone https://github.com/<TU_USUARIO>/OMR-SAM-2-Grader.git
%cd OMR-SAM-2-Grader
!pip install -r requirements.txt

## 🚀 Inicio Rápido en Google Colab

Si estás usando este proyecto en Google Colab, ejecuta estas celdas para configurar todo automáticamente:

```python
# 1. Clonar el repositorio
!git clone [https://github.com/josorio398/OMR-SAM2-Grader.git](https://github.com/josorio398/OMR-SAM2-Grader.git)
%cd OMR-SAM2-Grader

# 2. Configurar entorno y descargar modelos
!sh setup_colab.sh
!pip install -r requirements.txt -q

# 3. Ejecutar el calificador
!python main.py
