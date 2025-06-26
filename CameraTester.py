import cv2
from ultralytics import YOLO
from CameraCalibrator import CameraCalibrator

# Initialize YOLO and Calibrator
yolo = YOLO("yolo12s.pt")  # Replace with "yolov8n.pt" or similar if needed
calibrator = CameraCalibrator()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo(frame)
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": conf,
            "class": cls
        })
        # Draw box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

    # Run camera condition prediction
    condition_vec = calibrator.predict(frame, detections)

    # Display condition values on screen
    for i, val in enumerate(condition_vec[:15]):  # Display first 15 for space
        cv2.putText(frame, f"{i}: {val:.2f}", (10, 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Camera Tester", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
