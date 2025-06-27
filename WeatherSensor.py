import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torchvision.models import resnet18

class WeatherClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ImageEnhancementNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

class WeatherSensor:
    def __init__(self, device='cpu'):
        self.device = device
        self.weather_classes = ['sunny', 'cloudy', 'foggy', 'misty', 'rainy', 'stormy', 'snowy', 'lowlight', 'dusty']
        
        self.weather_classifier = WeatherClassifier(len(self.weather_classes)).to(device)
        self.enhancer = ImageEnhancementNN().to(device)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.weather_history = []
        self.history_length = 10
        
    def extract_weather_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        brightness = np.mean(gray)
        contrast = gray.std()
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_mean = np.mean(hsv[:, :, 1])
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_entropy = -np.sum((hist + 1e-8) * np.log2(hist + 1e-8))
        
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        rgb_mean = frame.mean(axis=(0, 1))
        rgb_std = frame.std(axis=(0, 1))
        
        features = np.array([
            brightness, contrast, edge_density, lap_var, saturation_mean,
            hist_entropy, blur_score, *rgb_mean, *rgb_std
        ])
        
        return features

    def classify_weather(self, frame):
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.weather_classifier(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        weather_condition = self.weather_classes[pred_idx]
        
        features = self.extract_weather_features(frame)
        
        if features[0] < 50 and features[1] < 30:
            weather_condition = 'lowlight'
        elif features[2] < 0.1 and features[3] < 100:
            weather_condition = 'foggy'
        elif features[1] < 20 and features[6] < 500:
            weather_condition = 'misty'
        
        self.weather_history.append(weather_condition)
        if len(self.weather_history) > self.history_length:
            self.weather_history.pop(0)
        
        stable_weather = max(set(self.weather_history), key=self.weather_history.count)
        
        return stable_weather, confidence

    def correct_sunny_conditions(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        corrected = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
        
        gamma = 0.8
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(corrected, table)
        
        return corrected

    def correct_cloudy_conditions(self, frame):
        alpha = 1.2
        beta = 20
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def correct_foggy_conditions(self, frame):
        frame_float = frame.astype(np.float32) / 255.0
        
        dark_channel = np.min(frame_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.morphologyEx(dark_channel, cv2.MORPH_CLOSE, kernel)
        
        A = np.max(frame_float)
        t = 1 - 0.95 * dark_channel / A
        t = np.maximum(t, 0.1)
        
        t = t[:, :, np.newaxis]
        dehazed = (frame_float - A) / t + A
        dehazed = np.clip(dehazed, 0, 1)
        
        result = (dehazed * 255).astype(np.uint8)
        
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        return result

    def correct_misty_conditions(self, frame):
        alpha = 1.3
        beta = 30
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.7, 0)
        
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        return result

    def correct_rainy_conditions(self, frame):
        alpha = 1.4
        beta = 25
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced

    def correct_stormy_conditions(self, frame):
        alpha = 1.6
        beta = 40
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 2.0, blurred, -1.0, 0)
        
        return sharpened

    def correct_snowy_conditions(self, frame):
        gamma = 0.7
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(frame, table)
        
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        corrected = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
        
        alpha = 1.3
        beta = -20
        corrected = cv2.convertScaleAbs(corrected, alpha=alpha, beta=beta)
        
        return corrected

    def correct_lowlight_conditions(self, frame):
        gamma = 0.4
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        brightened = cv2.LUT(frame, table)
        
        lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        alpha = 1.5
        beta = 30
        final = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
        
        return final

    def correct_dusty_conditions(self, frame):
        alpha = 1.3
        beta = 20
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.4)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
        
        return sharpened

    def neural_enhancement(self, frame):
        frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            enhanced = self.enhancer(frame_tensor)
            enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced

    def apply_corrections(self, frame, weather_condition):
        correction_map = {
            'sunny': self.correct_sunny_conditions,
            'cloudy': self.correct_cloudy_conditions,
            'foggy': self.correct_foggy_conditions,
            'misty': self.correct_misty_conditions,
            'rainy': self.correct_rainy_conditions,
            'stormy': self.correct_stormy_conditions,
            'snowy': self.correct_snowy_conditions,
            'lowlight': self.correct_lowlight_conditions,
            'dusty': self.correct_dusty_conditions
        }
        
        if weather_condition in correction_map:
            corrected = correction_map[weather_condition](frame)
        else:
            corrected = frame
        
        final_enhanced = self.neural_enhancement(corrected)
        
        return final_enhanced

    def process_frame(self, frame):
        weather_condition, confidence = self.classify_weather(frame)
        corrected_frame = self.apply_corrections(frame, weather_condition)
        
        return corrected_frame, weather_condition, confidence

def main():
    weather_sensor = WeatherSensor()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        corrected_frame, weather, confidence = weather_sensor.process_frame(frame)
        
        display_frame = np.hstack([frame, corrected_frame])
        
        cv2.putText(display_frame, f"Weather: {weather} ({confidence:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Original | Enhanced", 
                   (frame.shape[1]//2 - 100, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Weather Sensor - Original vs Enhanced", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()