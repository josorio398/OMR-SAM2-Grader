import os
import cv2
import numpy as np
import pandas as pd
import fitz
import torch
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM
from rapidocr_onnxruntime import RapidOCR

class OMRProcessor:
    def __init__(self):
        print("⏳ Cargando modelos de Inteligencia Artificial en memoria...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Carga IDÉNTICA a tu Colab original
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.sam_model = SAM("models/sam2_b.pt")
        self.ocr_engine = RapidOCR()
        
        os.makedirs("debug_antes", exist_ok=True)
        os.makedirs("debug_despues", exist_ok=True)
        print("✅ Modelos cargados exitosamente.\n")

    def process_pdf(self, pdf_filename):
        doc = fitz.open(pdf_filename)
        total_paginas = len(doc)
        print(f"\n📄 PDF cargado: '{pdf_filename}' con {total_paginas} páginas.")
        
        datos_salon = []

        # =========================================================
        # INICIO DEL BUCLE MAESTRO (IDÉNTICO AL ORIGINAL)
        # =========================================================
        for num_pagina in range(total_paginas):
            print(f"\n{'-'*50}")
            print(f"🚀 Procesando página {num_pagina + 1} de {total_paginas}...")
            print(f"{'-'*50}")

            try:
                # --- GESTIÓN DE MEMORIA (Añadido crucial para lotes) ---
                torch.cuda.empty_cache()
                import gc; gc.collect()
                # -------------------------------------------------------

                page = doc[num_pagina]
                pix = page.get_pixmap(dpi=300)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))

                # GUARDAMOS LA IMAGEN ORIGINAL INTACTA PARA EL OCR
                image_cv_raw = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # ====================================================================
                # 1. ENCUADRE SEGURO PARA DINO (AÍSLA LAS BURBUJAS)
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
                    if y > h_img * 0.90 and h < 100: continue
                    valid_coords.append((x, y, x+w, y+h))

                if valid_coords:
                    x_min = min([v[0] for v in valid_coords])
                    y_min = min([v[1] for v in valid_coords])
                    x_max = max([v[2] for v in valid_coords])
                    y_max = max([v[3] for v in valid_coords])

                    pad = 25
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(w_img, x_max + pad)
                    y_max = min(h_img, y_max + pad)

                    image_cv_recortada = image_cv_raw[y_min:y_max, x_min:x_max]
                else:
                    image_cv_recortada = image_cv_raw

                # ====================================================================
                # 2. ESTANDARIZACIÓN DE TAMAÑO (1600px) - SÓLO PARA DINO Y SAM
                # ====================================================================
                ancho_deseado = 1600
                escala = ancho_deseado / image_cv_recortada.shape[1]
                alto_deseado = int(image_cv_recortada.shape[0] * escala)
                image_cv = cv2.resize(image_cv_recortada, (ancho_deseado, alto_deseado), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(f"debug_antes/pagina_{num_pagina + 1}.jpg", image_cv)

                image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_cv_rgb)

                # --- Detección DINO ---
                text_prompt = "circle. bubble. black bubble. black spot. circle with letter inside"
                inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=0.06, text_threshold=0.06, target_sizes=[image_pil.size[::-1]]
                )[0]
                all_bboxes = results["boxes"].cpu().numpy().tolist()

                # --- Filtrado Estructural ---
                cajas_unicas = []
                for bbox in all_bboxes:
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    if w == 0 or h == 0: continue
                    es_duplicado = False
                    for u in cajas_unicas:
                        if abs(cx - u['cx']) < (image_pil.width * 0.015) and abs(cy - u['cy']) < (image_pil.width * 0.015):
                            es_duplicado = True
                            break
                    if not es_duplicado:
                        cajas_unicas.append({'bbox': bbox, 'cx': cx, 'cy': cy, 'area': w*h, 'ar': w/h})

                candidatas = [b for b in cajas_unicas if 0.4 <= b['ar'] <= 2.8 and b['area'] > 60]

                mid_x_image = image_pil.width / 2
                candidatas_izq = [b for b in candidatas if b['cx'] < mid_x_image]
                candidatas_der = [b for b in candidatas if b['cx'] >= mid_x_image]

                def filtrar_por_columnas(candidatos_mitad):
                    if not candidatos_mitad: return []
                    centros_x = sorted([b['cx'] for b in candidatos_mitad])
                    clusters = []
                    tol_x = image_pil.width * 0.04

                    for cx in centros_x:
                        if not clusters or cx - clusters[-1]['promedio'] > tol_x:
                            clusters.append({'valores': [cx], 'promedio': cx})
                        else:
                            clusters[-1]['valores'].append(cx)
                            clusters[-1]['promedio'] = sum(clusters[-1]['valores']) / len(clusters[-1]['valores'])

                    clusters_validos = [c for c in clusters if len(c['valores']) >= 3]
                    clusters_validos.sort(key=lambda c: c['promedio'])
                    columnas_respuestas = clusters_validos[-4:]

                    candidatos_finales = []
                    for b in candidatos_mitad:
                        distancias = [abs(b['cx'] - col['promedio']) for col in columnas_respuestas]
                        if distancias and min(distancias) <= (tol_x * 1.5):
                            candidatos_finales.append(b)
                    return candidatos_finales

                burbujas_izq = filtrar_por_columnas(candidatas_izq)
                burbujas_der = filtrar_por_columnas(candidatas_der)
                candidatas_filtradas = burbujas_izq + burbujas_der

                if len(candidatas_filtradas) < 80:
                    raise ValueError(f"DINO solo detectó {len(candidatas_filtradas)} burbujas tras filtrar.")

                mediana_area = np.median([b['area'] for b in candidatas_filtradas])
                candidatas_filtradas.sort(key=lambda b: abs(b['area'] - mediana_area))
                burbujas_finales = candidatas_filtradas[:80]

                col_izq_final = [b for b in burbujas_finales if b['cx'] < mid_x_image]
                col_der_final = [b for b in burbujas_finales if b['cx'] >= mid_x_image]

                def sort_column(col_data):
                    col_data.sort(key=lambda x: x['cy'])
                    sorted_final = []
                    for k in range(0, len(col_data), 4):
                        grupo = col_data[k:k+4]
                        grupo.sort(key=lambda x: x['cx'])
                        sorted_final.extend(grupo)
                    return sorted_final

                final_ordered_data = sort_column(col_izq_final) + sort_column(col_der_final)
                cajas_para_sam = [d['bbox'] for d in final_ordered_data]

                # --- SAM 2 ---
                sam_results = self.sam_model.predict(source=image_cv_rgb, bboxes=cajas_para_sam, save=False, verbose=False)
                mascaras_xy = sam_results[0].masks.xy if sam_results[0].masks is not None else []

                # 📸 DIAGNÓSTICO "DESPUÉS"
                viz_image = image_cv.copy()
                overlay = viz_image.copy()
                for poligono in mascaras_xy:
                    if len(poligono) > 0:
                        pts = np.array(poligono, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, viz_image, 0.6, 0, viz_image)
                cv2.imwrite(f"debug_despues/pagina_{num_pagina + 1}.jpg", viz_image)

                # --- Lógica Estadística de Intensidad ---
                image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                letras = ['A', 'B', 'C', 'D']
                resultados_examen = {}
                total_preguntas = min(20, len(mascaras_xy) // 4)

                for p in range(total_preguntas):
                    idx_inicio = p * 4
                    opciones_poligonos = mascaras_xy[idx_inicio : idx_inicio + 4]
                    promedios_intensidad = []
                    for poligono in opciones_poligonos:
                        mascara_vacia = np.zeros(image_gray.shape, dtype=np.uint8)
                        if len(poligono) > 0:
                            pts = np.array(poligono, np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mascara_vacia, [pts], 255)
                            pixeles_adentro = image_gray[mascara_vacia == 255]
                            promedio = np.mean(pixeles_adentro) if len(pixeles_adentro) > 0 else 255
                        else:
                            promedio = 255
                        promedios_intensidad.append(promedio)

                    min_val = np.min(promedios_intensidad)
                    max_val = np.max(promedios_intensidad)
                    diferencia = max_val - min_val

                    if diferencia < 45:
                        respuesta_final = "Sin respuesta"
                    else:
                        umbral_marcada = min_val + (diferencia * 0.35)
                        marcadas = [letras[j] for j, prom in enumerate(promedios_intensidad) if prom <= umbral_marcada]
                        respuesta_final = "Anulada" if len(marcadas) > 1 else marcadas[0]

                    resultados_examen[p + 1] = respuesta_final

                # ====================================================================
                # EXTRACCIÓN DE ENCABEZADO (BÚSQUEDA DIRECTA POR FORMA DE DATO)
                # ====================================================================
                alto_raw, ancho_raw = image_cv_raw.shape[:2]
                tercio_superior_raw = image_cv_raw[0:int(alto_raw * 0.35), :]

                resultados_ocr, _ = self.ocr_engine(tercio_superior_raw)
                texto_completo = " ".join([linea[1] for linea in resultados_ocr]) if resultados_ocr else ""

                # Limpiamos espacios dobles para facilitar la lectura matemática
                texto_limpio = re.sub(r'\s+', ' ', texto_completo)

                # 1. DOCUMENTO
                doc_match = re.search(r'\b(\d{8,12})\b', texto_limpio)

                # 2. CUADERNILLO
                cuad_match = re.search(r'\b([A-Za-z][_\-\s]\d{4}[_\-\s]\d[_\-\s]\d+)\b', texto_limpio)
                cuadernillo_final = "No detectado"
                if cuad_match:
                    cuadernillo_final = re.sub(r'[\-\s]', '_', cuad_match.group(1).upper())

                # 3. CURSO
                curso_match = None
                posibles_cursos = re.findall(r'\b([6-9]0[0-9]|1[01]0[0-9])\b', texto_limpio)
                if posibles_cursos:
                    curso_match = posibles_cursos[0]

                datos_encabezado = {
                    "Curso": curso_match if curso_match else "No detectado",
                    "Numero de Documento": doc_match.group(1) if doc_match else "No detectado",
                    "Numero de Cuadernillo": cuadernillo_final
                }

                # --- Estructuración de la Fila ---
                fila_estudiante = {
                    "Pagina_PDF": num_pagina + 1,
                    "Curso": datos_encabezado["Curso"],
                    "Numero de Documento": datos_encabezado["Numero de Documento"],
                    "Numero de Cuadernillo": datos_encabezado["Numero de Cuadernillo"]
                }
                for r in range(1, 21):
                    fila_estudiante[f"RP{r}"] = resultados_examen.get(r, "Sin datos")

                datos_salon.append(fila_estudiante)
                print(f"✅ Página {num_pagina + 1} procesada. (Curso: {datos_encabezado['Curso']} | Doc: {datos_encabezado['Numero de Documento']} | Cuad: {datos_encabezado['Numero de Cuadernillo']}).")

            except Exception as e:
                print(f"❌ Error en la página {num_pagina + 1}: {e}")
                datos_salon.append({
                    "Pagina_PDF": num_pagina + 1,
                    "Curso": "ERROR",
                    "Numero de Documento": "ERROR AL LEER",
                    "Numero de Cuadernillo": str(e)
                })

        output_name = "Resultados_Masivos_Salon.xlsx"
        pd.DataFrame(datos_salon).to_excel(output_name, index=False, engine='openpyxl')
        return output_name
