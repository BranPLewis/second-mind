import cv2
import os

# Create a folder for your data
os.makedirs("my_dataset", exist_ok=True)

# Open the camera (using GStreamer pipeline for Jetson)
cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)

count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow("Press SPACE to save, Q to quit", frame)
    
    key = cv2.waitKey(1)
    if key == ord(' '): # Spacebar
        img_name = f"my_dataset/image_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
