# 🎓 OMR SAM-2 Grader

Sistema inteligente de calificación de exámenes de selección múltiple basado en IA de vanguardia. Utiliza **Segment Anything Model 2 (SAM-2)** para una segmentación precisa y **Grounding DINO** para la detección de objetos sin entrenamiento previo.

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
