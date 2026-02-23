# Pipeline Diagram (placeholder)

```text
PDF (PyMuPDF @ 300dpi)
    ↓
Pre-encuadre (contornos + recorte)
    ↓
Resize (1600px ancho)
    ↓
Grounding DINO (detección zero-shot de burbujas)
    ↓
Filtrado estructural (duplicados / columnas / área)
    ↓
SAM-2 (segmentación por bboxes)
    ↓
Decisión por intensidad (A/B/C/D, Anulada, Sin respuesta)
    ↓
OCR encabezado (Curso / Documento / Cuadernillo)
    ↓
Exportación Excel + Debug (input/output + zip)
