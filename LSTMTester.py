import cv2
import numpy as np
from ultralytics import YOLO

class BaseYOLOv12:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def detect(self, image):
        results = self.model(image)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class": cls
            })
        return detections

class BridgingLayer:
    def adjust_boxes(self, detections):
        scaled_detections = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1
            scale = 1.3  
            nw, nh = int(w * scale), int(h * scale)
            new_box = [cx - nw//2, cy - nh//2, cx + nw//2, cy + nh//2]
            det["bbox"] = new_box
            scaled_detections.append(det)
        return scaled_detections

class AdaptiveFinalModel:
    def __init__(self):
        self.weight = 0.1  

    def update_weight(self):
        self.weight = min(1.0, self.weight + 0.02)

    def predict(self, adjusted_boxes):
        self.update_weight()
        return adjusted_boxes

class LSTMController:
    def __init__(self):
        self.base_model = BaseYOLOv12()
        self.bridge = BridgingLayer()
        self.final_model = AdaptiveFinalModel()

    def run(self, frame):
        
        output = frame.copy()

        yolo_boxes = self.base_model.detect(frame)
        bridged_boxes = self.bridge.adjust_boxes(yolo_boxes)
        final_boxes = self.final_model.predict(bridged_boxes)
        class_names = self.base_model.model.names

       
        for box in yolo_boxes:
            x1, y1, x2, y2 = box["bbox"]
            cls = box["class"]
            conf = box["confidence"]
            name = class_names.get(cls, str(cls))
            label = f"YOLO: {name} {conf:.2f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    
        for box in final_boxes:
            x1, y1, x2, y2 = box["bbox"]
            cls = box["class"]
            conf = box["confidence"]
            name = class_names.get(cls, str(cls))
            green_intensity = int(255 * self.final_model.weight)
            label = f"Final: {name} {conf:.2f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, green_intensity, 0), 2)
            cv2.putText(output, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, green_intensity, 0), 2)

        cv2.putText(output, f"Final Model Influence: {self.final_model.weight:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        return output


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    controller = LSTMController()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame = controller.run(frame)
        cv2.imshow("YOLOv12 LSTM Fusion", out_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
