import torch
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class OMRDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Carga de DINO
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        # Carga de SAM-2
        self.sam_model = SAM("models/sam2_b.pt")

    def detect_bubbles(self, image_pil):
        text_prompt = "circle. bubble. black bubble. black spot."
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.06, text_threshold=0.06, 
            target_sizes=[image_pil.size[::-1]]
        )[0]
        return results["boxes"].cpu().numpy().tolist()

    def segment_with_sam(self, image_rgb, boxes):
        return self.sam_model.predict(source=image_rgb, bboxes=boxes, save=False, verbose=False)
