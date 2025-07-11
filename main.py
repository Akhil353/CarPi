import cv2
from CAM import CameraMonitor

monitor = CameraMonitor()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    is_drift, dist = monitor.process_frame(frame)
    print(f"Drift: {is_drift}, Distance: {dist:.2f}")

    if is_drift:
        overlay = monitor.get_blindspot_map(frame)
        cv2.imshow("Blindspot Map", overlay)

    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
