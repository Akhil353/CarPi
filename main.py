import cv2
import numpy as np
from WeatherSensor import WeatherSensor

def main():
    # Use GPU if available
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    weather_sensor = WeatherSensor(device=device)

    # Open webcam (or replace with video file path)
    cap = cv2.VideoCapture(0)

    # Set resolution and FPS (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through weather pipeline
        corrected_frame, weather, confidence = weather_sensor.process_frame(frame)

        # Combine original and corrected side-by-side
        display_frame = np.hstack([frame, corrected_frame])

        # Draw weather condition
        cv2.putText(display_frame, f"Weather: {weather} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display_frame, "Original | Enhanced",
                    (frame.shape[1] // 2 - 100, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the final output
        cv2.imshow("Weather Sensor - Original vs Enhanced", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
