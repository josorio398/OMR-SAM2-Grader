import cv2
import numpy as np
import pandas as pd
import fitz
from .detector import OMRDetector
from .ocr_engine import OMROCR

class OMRProcessor:
    def __init__(self):
        print("⏳ Inicializando modelos de IA...")
        self.detector = OMRDetector()
        self.ocr = OMROCR()

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        datos_finales = []
        
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            
            # 1. Extraer encabezado con OCR
            header_img = img[0:int(img.shape[0] * 0.35), :]
            header_data = self.ocr.extract_header(header_img)
            
            # 2. Aquí integrarías el bucle de calificación que ya tienes...
            # (Por brevedad, guardamos los datos del encabezado)
            
            fila = {
                "Pagina": i + 1,
                "Documento": header_data["documento"],
                "Cuadernillo": header_data["cuadernillo"]
            }
            datos_finales.append(fila)
            print(f"✅ Página {i+1} procesada.")

        output = "Resultados_Finales.xlsx"
        pd.DataFrame(datos_finales).to_excel(output, index=False)
        return output
