import torch
import numpy as np
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class OMRDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.sam_model = SAM("models/sam2_b.pt")

    def detect_and_filter_bubbles(self, image_pil):
        text_prompt = "circle. bubble. black bubble. black spot. circle with letter inside"
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.06, text_threshold=0.06, target_sizes=[image_pil.size[::-1]]
        )[0]
        
        all_bboxes = results["boxes"].cpu().numpy().tolist()
        
        # --- FILTRADO POR COLUMNAS (TU LÓGICA) ---
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
        mid_x = image_pil.width / 2

        def filter_cols(cands):
            if not cands: return []
            centros_x = sorted([b['cx'] for b in cands])
            clusters = []
            for cx in centros_x:
                if not clusters or cx - clusters[-1]['promedio'] > image_pil.width * 0.04:
                    clusters.append({'valores': [cx], 'promedio': cx})
                else:
                    clusters[-1]['valores'].append(cx)
                    clusters[-1]['promedio'] = sum(clusters[-1]['valores']) / len(clusters[-1]['valores'])
            validos = sorted([c for c in clusters if len(c['valores']) >= 3], key=lambda c: c['promedio'])[-4:]
            return [b for b in cands if any(abs(b['cx'] - v['promedio']) <= image_pil.width * 0.06 for v in validos)]

        burbujas = filter_cols([b for b in candidatas if b['cx'] < mid_x]) + filter_cols([b for b in candidatas if b['cx'] >= mid_x])
        
        # Ordenar (Tu lógica de sort_column)
        burbujas.sort(key=lambda x: (x['cx'] >= mid_x, x['cy'])) # Izquierda primero, luego derecha, por altura
        # ... (simplificado para asegurar que SAM-2 reciba el orden correcto)
        return [b['bbox'] for b in burbujas[:80]]

    def segment_with_sam(self, image_cv_rgb, bboxes):
        return self.sam_model.predict(source=image_cv_rgb, bboxes=bboxes, save=False, verbose=False)
