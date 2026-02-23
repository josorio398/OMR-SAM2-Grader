import re
from rapidocr_onnxruntime import RapidOCR

class OMROCR:
    def __init__(self):
        self.engine = RapidOCR()

    def extract_header(self, image_np):
        results, _ = self.engine(image_np)
        texto_completo = " ".join([linea[1] for linea in results]) if results else ""
        texto_limpio = re.sub(r'\s+', ' ', texto_completo)

        # Regex para Documento, Cuadernillo y Curso
        doc_match = re.search(r'\b(\d{8,12})\b', texto_limpio)
        cuad_match = re.search(r'\b([A-Za-z][_\-\s]\d{4}[_\-\s]\d[_\-\s]\d+)\b', texto_limpio)
        
        return {
            "documento": doc_match.group(1) if doc_match else "No detectado",
            "cuadernillo": re.sub(r'[\-\s]', '_', cuad_match.group(1).upper()) if cuad_match else "No detectado",
            "texto_crudo": texto_limpio
        }
