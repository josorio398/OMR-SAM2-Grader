import re
import cv2
from rapidocr_onnxruntime import RapidOCR

class OMROCR:
    def __init__(self):
        self.engine = RapidOCR()

    def extract_header(self, image_cv_raw):
        alto, ancho = image_cv_raw.shape[:2]
        tercio = image_cv_raw[0:int(alto * 0.35), :]
        results, _ = self.engine(tercio)
        texto = " ".join([linea[1] for linea in results]) if results else ""
        texto = re.sub(r'\s+', ' ', texto)

        doc = re.search(r'\b(\d{8,12})\b', texto)
        cuad = re.search(r'\b([A-Za-z][_\-\s]\d{4}[_\-\s]\d[_\-\s]\d+)\b', texto)
        curso = re.findall(r'\b([6-9]0[0-9]|1[01]0[0-9])\b', texto)

        return {
            "Curso": curso[0] if curso else "No detectado",
            "Numero de Documento": doc.group(1) if doc else "No detectado",
            "Numero de Cuadernillo": re.sub(r'[\-\s]', '_', cuad.group(1).upper()) if cuad else "No detectado"
        }
