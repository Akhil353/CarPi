import cv2
from ultralytics import YOLO
from CameraCalibrator import CameraCalibrator

# Initialize YOLO and Calibrator
yolo = YOLO("yolo12s.pt")  # Replace with your model path
calibrator = CameraCalibrator()

# Open webcam
cap = cv2.VideoCapture(0)

# Label mapping for better readability
LABELS = [
    "RGB Mean R", "RGB Mean G", "RGB Mean B",
    "RGB Std R", "RGB Std G", "RGB Std B",
    "Brightness", "Contrast", "Focus (Laplacian)",
    "Tenengrad", "FFT Blur", "Entropy",
    "Edge Magnitude", "Gamma", "Skew"
]

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
        # Draw detection box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

    # Run calibration check
    results = calibrator.predict(frame, detections)
    prediction = results["prediction"]
    tint_hex = results["tint_hex"]
    blindspots = results["blind_grid_coords"]

    # Display first 15 interpreted prediction values
    for i, label in enumerate(LABELS):
        val = prediction[i]
        cv2.putText(frame, f"{label}: {val:.2f}", (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display tint color
    cv2.putText(frame, f"Approx Tint: {tint_hex}", (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (int(results['avg_rgb'][2]), int(results['avg_rgb'][1]), int(results['avg_rgb'][0])), 2)

    # Highlight blind grid cells
    h, w, _ = frame.shape
    grid_size = 6
    gh, gw = h // grid_size, w // grid_size
    for i, j in blindspots:
        x1, y1 = j * gw, i * gh
        x2, y2 = x1 + gw, y1 + gh
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Blind", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Camera Tester", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
