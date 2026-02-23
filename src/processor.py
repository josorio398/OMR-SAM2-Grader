import cv2
import numpy as np
import pandas as pd
import fitz
import os
from .detector import OMRDetector
from .ocr_engine import OMROCR

class OMRProcessor:
    def __init__(self):
        print("⏳ Cargando modelos de Inteligencia Artificial en memoria...")
        self.detector = OMRDetector()
        self.ocr = OMROCR()
        
        # Preparar carpetas de diagnóstico
        os.makedirs("debug/antes", exist_ok=True)
        os.makedirs("debug/despues", exist_ok=True)

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        datos_finales = []
        letras = ['A', 'B', 'C', 'D']
        total_paginas = len(doc)

        print(f"\n📄 PDF cargado: '{pdf_path}' con {total_paginas} páginas.")

        for i in range(total_paginas):
            print(f"\n{'-'*50}")
            print(f"🚀 Procesando página {i + 1} de {total_paginas}...")
            print(f"{'-'*50}")

            try:
                page = doc[i]
                pix = page.get_pixmap(dpi=300)
                # Conversión de colores para OpenCV y PIL
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                
                # Guardar imagen original para diagnóstico "antes"
                cv2.imwrite(f"debug/antes/pagina_{i+1}.jpg", img_bgr)

                # 1. Extracción de Encabezado (OCR)
                # Tomamos el tercio superior para el OCR
                alto, ancho = img_bgr.shape[:2]
                header_img = img_bgr[0:int(alto * 0.35), :]
                header_data = self.ocr.extract_header(header_img)

                # 2. Detección y Segmentación (IA)
                from PIL import Image
                img_pil = Image.fromarray(img_rgb)
                bboxes = self.detector.detect_bubbles(img_pil)
                
                # Aquí podrías añadir lógica de filtrado/ordenado de bboxes si fuera necesario
                # Por ahora usamos los resultados de SAM-2 directamente
                sam_results = self.detector.segment_with_sam(img_rgb, bboxes)
                mascaras_xy = sam_results[0].masks.xy if sam_results[0].masks is not None else []

                # --- Generar imagen de diagnóstico "después" ---
                viz_image = img_bgr.copy()
                overlay = viz_image.copy()
                for poligono in mascaras_xy:
                    if len(poligono) > 0:
                        pts = np.array(poligono, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, viz_image, 0.6, 0, viz_image)
                cv2.imwrite(f"debug/despues/pagina_{i+1}.jpg", viz_image)

                # 3. Bucle de Calificación por Intensidad
                respuestas_pag = {}
                # Suponemos 20 preguntas (80 burbujas)
                total_preguntas = min(20, len(mascaras_xy) // 4)

                for p in range(total_preguntas):
                    idx = p * 4
                    opciones_poligonos = mascaras_xy[idx : idx + 4]
                    promedios = []

                    for poligono in opciones_poligonos:
                        mask = np.zeros(img_gray.shape, dtype=np.uint8)
                        if len(poligono) > 0:
                            pts = np.array(poligono, np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [pts], 255)
                            pixeles_adentro = img_gray[mask == 255]
                            promedio = np.mean(pixeles_adentro) if len(pixeles_adentro) > 0 else 255
                        else:
                            promedio = 255
                        promedios.append(promedio)

                    # Lógica de decisión
                    min_v, max_v = np.min(promedios), np.max(promedios)
                    dif = max_v - min_v

                    if dif < 45:
                        res = "Sin respuesta"
                    else:
                        umbral = min_v + (dif * 0.35)
                        marcadas = [letras[j] for j, prom in enumerate(promedios) if prom <= umbral]
                        res = "Anulada" if len(marcadas) > 1 else (marcadas[0] if marcadas else "Sin respuesta")
                    
                    respuestas_pag[f"RP{p+1}"] = res

                # 4. Consolidar Fila
                fila = {
                    "Pagina_PDF": i + 1,
                    "Curso": header_data.get("documento")[:3] if header_data["documento"] != "No detectado" else "Error",
                    "Numero de Documento": header_data["documento"],
                    "Numero de Cuadernillo": header_data["cuadernillo"]
                }
                fila.update(respuestas_pag)
                datos_finales.append(fila)

                print(f"✅ Página {i + 1} procesada. (Doc: {fila['Numero de Documento']} | Cuad: {fila['Numero de Cuadernillo']}).")

            except Exception as e:
                print(f"❌ Error en la página {i + 1}: {e}")
                datos_finales.append({"Pagina_PDF": i+1, "Curso": "ERROR", "Numero de Documento": str(e)})

        # 5. Exportación Final
        output = "Resultados_Masivos_Salon.xlsx"
        df = pd.DataFrame(datos_finales)
        df.to_excel(output, index=False)
        return output
