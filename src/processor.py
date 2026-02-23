import cv2
import numpy as np
import pandas as pd
import fitz
import os
import torch  # <--- Importante para limpiar memoria

class OMRProcessor:
    def __init__(self):
        print("⏳ Cargando modelos de Inteligencia Artificial...")
        from .detector import OMRDetector
        from .ocr_engine import OMROCR
        self.detector = OMRDetector()
        self.ocr = OMROCR()
        os.makedirs("debug/antes", exist_ok=True)
        os.makedirs("debug/despues", exist_ok=True)

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        datos_finales = []
        letras = ['A', 'B', 'C', 'D']
        total_paginas = len(doc)

        for i in range(total_paginas):
            print(f"\n{'-'*50}\n🚀 Procesando página {i + 1} de {total_paginas}...\n{'-'*50}")

            try:
                page = doc[i]
                pix = page.get_pixmap(dpi=300)
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # 1. OCR
                header_img = img_bgr[0:int(img_bgr.shape[0] * 0.35), :]
                header_data = self.ocr.extract_header(header_img)

                # 2. IA (DINO + SAM)
                from PIL import Image
                img_pil = Image.fromarray(img_rgb)
                bboxes = self.detector.detect_bubbles(img_pil)
                sam_results = self.detector.segment_with_sam(img_rgb, bboxes)
                mascaras_xy = sam_results[0].masks.xy if sam_results[0].masks is not None else []

                # 3. Calificación por intensidad
                respuestas_pag = {}
                total_preguntas = min(20, len(mascaras_xy) // 4)
                for p in range(total_preguntas):
                    idx = p * 4
                    opciones = mascaras_xy[idx : idx + 4]
                    promedios = []
                    for poligono in opciones:
                        mask = np.zeros(img_gray.shape, dtype=np.uint8)
                        if len(poligono) > 0:
                            pts = np.array(poligono, np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [pts], 255)
                            pixeles = img_gray[mask == 255]
                            promedios.append(np.mean(pixeles) if len(pixeles) > 0 else 255)
                        else: promedios.append(255)
                    
                    min_v, max_v = np.min(promedios), np.max(promedios)
                    if (max_v - min_v) < 45: res = "Sin respuesta"
                    else:
                        umbral = min_v + ((max_v - min_v) * 0.35)
                        marcadas = [letras[j] for j, prom in enumerate(promedios) if prom <= umbral]
                        res = "Anulada" if len(marcadas) > 1 else (marcadas[0] if marcadas else "Sin respuesta")
                    respuestas_pag[f"RP{p+1}"] = res

                fila = {
                    "Pagina_PDF": i + 1,
                    "Curso": header_data["documento"][:3] if header_data["documento"] != "No detectado" else "Error",
                    "Numero de Documento": header_data["documento"],
                    "Numero de Cuadernillo": header_data["cuadernillo"]
                }
                fila.update(respuestas_pag)
                datos_finales.append(fila)
                print(f"✅ Página {i + 1} lista. (Doc: {fila['Numero de Documento']})")

            except Exception as e:
                print(f"❌ Error en la página {i + 1}: {e}")
                datos_finales.append({"Pagina_PDF": i+1, "Curso": "ERROR", "Numero de Documento": str(e)})
            
            # ==========================================
            # 🔥 LA CLAVE: LIMPIEZA DE MEMORIA GPU
            # ==========================================
            torch.cuda.empty_cache() 
            import gc
            gc.collect()
            # ==========================================

        output = "Resultados_Masivos_Salon.xlsx"
        pd.DataFrame(datos_finales).to_excel(output, index=False)
        return output
