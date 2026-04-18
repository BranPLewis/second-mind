import base64
import cv2
import numpy as np
import modal
import os

app = modal.App("second-mind")

# 1. Chain `.add_local_dir` directly to the image to securely upload your project files
vision_image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "numpy",
        "fastapi[standard]",
        "ultralytics" 
    )
    .add_local_dir(".", remote_path="/root/project", ignore=[".venv", "__pycache__", ".gitignore", ".git"]) # <-- NEW 1.0 SYNTAX
)

# 2. The `mounts` argument has been removed from the decorator
@app.cls(image=vision_image, gpu="T4")
class ModelInference:
    
    @modal.enter()
    def load_model(self):
        # This function runs ONCE when the container wakes up.
        import sys
        sys.path.append("/root/project")
        
        # Import your custom detector from your local vision.py file
        from vision import ObjectDetector
        
        # NOTE: Update "best.pt" to whatever your YOLO weights file is actually named
        model_path = os.getenv("YOLO_MODEL_PATH", "/root/project/model/weights/best.pt")
        
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence=float(os.getenv("YOLO_CONFIDENCE", "0.05"))
        )

    @modal.fastapi_endpoint(method="POST")
    def process_image(self, data: dict):
        b64_image = data.get("image")
        if not b64_image:
            return {"error": "No image provided."}

        # --- A. Decode the Base64 image back into a usable format ---
        raw_bytes = base64.b64decode(b64_image)
        np_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image format."}

        # --- B. Run the detection logic using the pre-loaded model ---
        detections, _ = self.detector.detect_components(image)
        label_color_map = self.detector.build_label_color_map([d.label for d in detections])
        annotated = self.detector.draw_all_detections(
            image,
            detections,
            color_map=label_color_map,
        )

        # --- C. Encode the newly annotated image back to Base64 ---
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            return {"error": "Could not render annotated image."}
        annotated_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        if not detections:
            return {
                "detected": False,
                "message": "No components detected in this image.",
                "components": [],
                "total_detections": 0,
                "annotated_image": annotated_b64,
            }

        # --- D. Format the metadata exactly how your frontend expects it ---
        best_by_label = self.detector.highest_confidence_per_label(detections)
        counts_by_label = {}
        for detection in detections:
            counts_by_label[detection.label] = counts_by_label.get(detection.label, 0) + 1

        components = []
        for label, best in best_by_label.items():
            color = self.detector.bgr_to_hex(label_color_map[label]) if label in label_color_map else self.detector.label_color_hex(label)
            components.append({
                "label": label,
                "count": counts_by_label[label],
                "confidence": round(best.confidence, 4),
                "color": color,
                "center": {"x": best.center[0], "y": best.center[1]},
                "bbox": {
                    "x1": best.bbox[0],
                    "y1": best.bbox[1],
                    "x2": best.bbox[2],
                    "y2": best.bbox[3],
                },
            })

        components.sort(key=lambda comp: comp["confidence"], reverse=True)

        return {
            "detected": True,
            "total_detections": len(detections),
            "components": components,
            "annotated_image": annotated_b64
        }
