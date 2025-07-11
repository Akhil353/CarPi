import cv2
import numpy as np
import torch
import time
from WeatherSensor import WeatherSensor
from CAM import CameraMonitor

# ─── YOLOv5 LOADER ───────────────────────────────────────
from pathlib import Path
import sys
sys.path.append(str(Path("yolov5")))  # Path to YOLOv5 folder
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

class YOLOWrapper:
    def __init__(self, weights="yolov5s.pt", device='cpu', imgsz=416):
        self.device = device
        self.model = DetectMultiBackend(weights, device=self.device)
        self.model.eval()
        self.imgsz = imgsz
        self.names = self.model.names

    def preprocess(self, img):
        img_resized = letterbox(img, self.imgsz, stride=32, auto=True)[0]
        img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float()
        img_tensor /= 255.0
        return img_tensor.unsqueeze(0).to(self.device), img_resized.shape[:2]

    def detect(self, frame):
        img_tensor, shape = self.preprocess(frame)
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)[0]

        dets = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred:
                label = f"{self.names[int(cls)]} {conf:.2f}"
                dets.append((xyxy, label))
        return dets

    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        for (x1, y1, x2, y2), label in detections:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

# ─── BLINDSPOT PATCH ENHANCEMENT ─────────────────────────
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

# ─── MAIN COMBINED LOOP ───────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weather_sensor = WeatherSensor(device=device)
    camera_monitor = CameraMonitor(device=device)
    yolo = YOLOWrapper(device=device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_skip = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_skip += 1
        if frame_skip % 2 != 0:
            continue  # skip every other frame for speed

        # Step 1: Camera blindspot detection
        _, _, blindspots = camera_monitor.process_frame(frame)
        blindspot_fixed = enhance_frame_with_blindspots(frame, blindspots)

        # Step 2: Weather correction
        enhanced, weather, conf = weather_sensor.process_frame(blindspot_fixed)

        # Step 3: YOLO detection on both frames
        det_orig = yolo.detect(frame)
        det_enhanced = yolo.detect(enhanced)

        frame_detected = yolo.draw_detections(frame.copy(), det_orig, (0, 0, 255))
        enhanced_detected = yolo.draw_detections(enhanced.copy(), det_enhanced, (0, 255, 0))

        # Step 4: Display side-by-side
        display_frame = np.hstack([frame_detected, enhanced_detected])
        cv2.putText(display_frame, f"Weather: {weather} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Original (RED) | Enhanced (GREEN)",
                    (frame.shape[1] // 2 - 100, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Object Detection Comparison", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
