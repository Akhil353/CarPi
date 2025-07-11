import cv2
import numpy as np
import torch
from ultralytics import YOLO
from WeatherSensor import WeatherSensor
from CAM import CameraMonitor  # Your camera enhancement module

def enhance_frame_with_blindspots(frame, blindspots):
    sharpened = frame.copy()
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    for spot in blindspots:
        x1, y1, x2, y2 = spot["bbox"]
        roi = sharpened[y1:y2, x1:x2]
        sharpened_roi = cv2.filter2D(roi, -1, kernel)
        sharpened[y1:y2, x1:x2] = sharpened_roi
    return sharpened

def draw_yolo(model, frame, imgsz=416):
    results = model(frame, imgsz=imgsz, verbose=False)[0]
    return results.plot()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weather_sensor = WeatherSensor(device=device)
    camera_monitor = CameraMonitor(device=device)

    model = YOLO("yolo12s.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    skip_every = 1  # increase for more speed, e.g. 2 skips every other frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_every != 0:
            continue

        # Step 1: Detect blindspots (skip retraining/logging)
        _, _, blindspots = camera_monitor.process_frame(frame)

        # Step 2: Sharpen blurry regions
        sharpened = enhance_frame_with_blindspots(frame, blindspots)

        # Step 3: Weather correction
        enhanced, weather, confidence = weather_sensor.process_frame(sharpened)

        # Step 4: Run YOLO on both
        yolo_orig = draw_yolo(model, frame)
        yolo_enhanced = draw_yolo(model, enhanced)

        # Combine and annotate
        combined = np.hstack([yolo_orig, yolo_enhanced])
        cv2.putText(combined, f"Weather: {weather} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(combined, "Original | Enhanced", (frame.shape[1] - 150, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("YOLOv12 Detection | Camera + Weather Enhancement", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
