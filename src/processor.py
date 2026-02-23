import cv2
import numpy as np
import pandas as pd
import fitz
import os
import torch
import gc
import re
from PIL import Image

class OMRProcessor:
    def __init__(self):
        print("⏳ Cargando modelos de Inteligencia Artificial...")
        from .detector import OMRDetector
        from .ocr_engine import OMROCR
        self.detector = OMRDetector()
        self.ocr = OMROCR()
        os.makedirs("debug_antes", exist_ok=True)
        os.makedirs("debug_despues", exist_ok=True)

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        datos_salon = []
        letras = ['A', 'B', 'C', 'D']
        total_paginas = len(doc)

        for num_pagina in range(total_paginas):
            print(f"\n{'-'*50}\n🚀 Procesando página {num_pagina + 1} de {total_paginas}...\n{'-'*50}")

            try:
                # LIMPIEZA PREVENTIVA DE MEMORIA
                torch.cuda.empty_cache()
                gc.collect()

                page = doc[num_pagina]
                pix = page.get_pixmap(dpi=300)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
                image_cv_raw = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # ====================================================================
                # 1. ENCUADRE SEGURO (TU LÓGICA GANADORA)
                # ====================================================================
                gray = cv2.cvtColor(image_cv_raw, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
                cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                valid_coords = []
                h_img, w_img = image_cv_raw.shape[:2]
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    if w < 15 or h < 15: continue
                    if w > w_img * 0.95 or h > h_img * 0.95: continue
                    valid_coords.append((x, y, x+w, y+h))

                if valid_coords:
                    x_min, y_min = min([v[0] for v in valid_coords]), min([v[1] for v in valid_coords])
                    x_max, y_max = max([v[2] for v in valid_coords]), max([v[3] for v in valid_coords])
                    pad = 25
                    image_cv_recortada = image_cv_raw[max(0, y_min-pad):min(h_img, y_max+pad), max(0, x_min-pad):min(w_img, x_max+pad)]
                else:
                    image_cv_recortada = image_cv_raw

                # ====================================================================
                # 2. REDIMENSIÓN A 1600px (CLAVE PARA LA MEMORIA)
                # ====================================================================
                ancho_deseado = 1600
                escala = ancho_deseado / image_cv_recortada.shape[1]
                alto_deseado = int(image_cv_recortada.shape[0] * escala)
                image_cv = cv2.resize(image_cv_recortada, (ancho_deseado, alto_deseado), interpolation=cv2.INTER_CUBIC)
                
                cv2.imwrite(f"debug_antes/pagina_{num_pagina + 1}.jpg", image_cv)
                image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_cv_rgb)

                # 3. DETECCIÓN Y SEGMENTACIÓN
                with torch.no_grad():
                    # Llamamos a tu lógica de filtrado compleja que está en detector.py
                    bboxes = self.detector.detect_and_filter_bubbles(image_pil)
                    sam_results = self.detector.segment_with_sam(image_cv_rgb, bboxes)
                
                mascaras_xy = sam_results[0].masks.xy if sam_results[0].masks is not None else []

                # --- DIAGNÓSTICO DESPUÉS ---
                viz_image = image_cv.copy()
                overlay = viz_image.copy()
                for poligono in mascaras_xy:
                    if len(poligono) > 0:
                        pts = np.array(poligono, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, viz_image, 0.6, 0, viz_image)
                cv2.imwrite(f"debug_despues/pagina_{num_pagina + 1}.jpg", viz_image)

                # 4. CALIFICACIÓN POR INTENSIDAD
                image_gray_proc = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                resultados_examen = {}
                total_preguntas = min(20, len(mascaras_xy) // 4)

                for p in range(total_preguntas):
                    idx = p * 4
                    opciones = mascaras_xy[idx : idx + 4]
                    promedios = []
                    for poligono in opciones:
                        mask = np.zeros(image_gray_proc.shape, dtype=np.uint8)
                        if len(poligono) > 0:
                            pts = np.array(poligono, np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [pts], 255)
                            pixeles = image_gray_proc[mask == 255]
                            promedios.append(np.mean(pixeles) if len(pixeles) > 0 else 255)
                        else: promedios.append(255)
                    
                    min_v, max_v = np.min(promedios), np.max(promedios)
                    dif = max_v - min_v
                    if dif < 45: res = "Sin respuesta"
                    else:
                        umbral = min_v + (dif * 0.35)
                        marcadas = [letras[j] for j, p_val in enumerate(promedios) if p_val <= umbral]
                        res = "Anulada" if len(marcadas) > 1 else (marcadas[0] if marcadas else "Sin respuesta")
                    resultados_examen[p + 1] = res

                # 5. OCR DE ENCABEZADO
                header_data = self.ocr.extract_header(image_cv_raw)

                # CONSOLIDAR FILA
                fila = {
                    "Pagina_PDF": num_pagina + 1,
                    "Curso": header_data["Curso"],
                    "Numero de Documento": header_data["Numero de Documento"],
                    "Numero de Cuadernillo": header_data["Numero de Cuadernillo"]
                }
                for i in range(1, 21):
                    fila[f"RP{i}"] = resultados_examen.get(i, "Sin datos")
                
                datos_salon.append(fila)
                print(f"✅ Página {num_pagina + 1} procesada. (Curso: {fila['Curso']} | Doc: {fila['Numero de Documento']}).")

            except Exception as e:
                print(f"❌ Error en la página {num_pagina + 1}: {e}")
                datos_salon.append({"Pagina_PDF": num_pagina + 1, "Curso": "ERROR", "Numero de Documento": str(e)})

        # EXPORTACIÓN
        output = "Resultados_Masivos_Salon.xlsx"
        pd.DataFrame(datos_salon).to_excel(output, index=False)
        return output
