# Demo (assets)

Este folder contiene archivos de prueba para validar el pipeline.

## 📄 sample_4pages.pdf
- PDF con **4 evaluaciones** (1 página = 1 estudiante).
- Formato: *Colegio Ciudad de Villavicencio IED – Hoja de Respuesta*.

### Cómo usarlo en Colab
1. Ejecuta el notebook:
   `notebooks/OMR_SAM2_Grader_Colab.ipynb`
2. En la etapa de ejecución, cuando aparezca:
   **"Sube el archivo PDF con las hojas de respuestas..."**
3. Sube el archivo `assets/demo/sample_4pages.pdf`

### Salidas esperadas
- `outputs/Resultados_Masivos_Salon.xlsx`
- `outputs/Imagenes_Diagnostico.zip`
- `debug/input/` y `debug/output/` con imágenes por página
