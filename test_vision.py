import os
from datetime import datetime

import cv2
from ultralytics.models.yolo.model import YOLO


MODEL_PATH = "runs/detect/train5/weights/best.pt"
OUTPUT_DIR = "captures"


def build_jetson_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def print_detections(result):
    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
        return

    print("Detections:")
    for i, box in enumerate(result.boxes, start=1):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = result.names.get(cls_id, str(cls_id))
        print(f"  {i}. {label} ({conf:.2f})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("Opening camera...")
    cap = cv2.VideoCapture(build_jetson_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Jetson pipeline failed, falling back to default camera index 0...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    window_name = "Live Camera - SPACE: capture+infer | Q: quit"
    print("Camera ready. Press SPACE to capture and run inference, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Exiting.")
            break

        if key == ord(" "):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(OUTPUT_DIR, f"capture_{timestamp}.jpg")
            out_path = os.path.join(OUTPUT_DIR, f"capture_{timestamp}_pred.jpg")

            cv2.imwrite(raw_path, frame)
            print(f"Captured frame: {raw_path}")
            print("Running model inference...")

            results = model(frame)
            result = results[0]
            annotated_frame = result.plot()

            cv2.imwrite(out_path, annotated_frame)
            print(f"Saved prediction image: {out_path}")
            print_detections(result)

            cv2.imshow("Prediction", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
