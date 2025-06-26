import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision.models import resnet18
from urllib.request import urlretrieve
import os
import random

class CameraConditionNN(nn.Module):
    def __init__(self, input_dim=128, output_dim=30):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class CameraCalibrator:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = CameraConditionNN().to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Pretrained ResNet as feature extractor
        resnet = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor()
        ])

        self.reference_images = self._load_reference_samples()

    def _load_reference_samples(self):
        ref_dir = "calibrator_refs"
        os.makedirs(ref_dir, exist_ok=True)

        refs = {}
        # Only a few common class IDs from COCO: person(0), car(2), dog(16), etc.
        urls = {
            0: "https://ultralytics.com/images/bus.jpg",
            2: "https://ultralytics.com/images/zidane.jpg",
            16: "https://ultralytics.com/images/bus.jpg"
        }

        for cls, url in urls.items():
            path = os.path.join(ref_dir, f"{cls}.jpg")
            if not os.path.exists(path):
                urlretrieve(url, path)
            refs[cls] = cv2.imread(path)

        return refs

    def _visual_stats(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_mean = img.mean(axis=(0, 1))
        rgb_std = img.std(axis=(0, 1))
        brightness = np.mean(gray)
        contrast = gray.std()
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        entropy = -np.sum((p := np.histogram(gray, bins=256)[0]/(gray.size + 1e-8)) * np.log2(p + 1e-8))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edge_mag = np.sqrt(sobelx**2 + sobely**2).mean()
        gamma = np.mean(np.power(img / 255.0 + 1e-8, 2.2))
        skew = ((gray - gray.mean())**3).mean() / (gray.std()**3 + 1e-8)
        kurtosis = ((gray - gray.mean())**4).mean() / (gray.std()**4 + 1e-8)
        return list(rgb_mean) + list(rgb_std) + [brightness, contrast, lap_var, entropy, edge_mag, gamma, skew, kurtosis]

    def _feature_vector(self, crop, ref_crop):
        if crop.size == 0 or ref_crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        diff_stats = np.array(self._visual_stats(crop)) - np.array(self._visual_stats(ref_crop))

        crop_t = self.transform(crop).unsqueeze(0)
        ref_t = self.transform(ref_crop).unsqueeze(0)
        with torch.no_grad():
            f1 = self.feature_extractor(crop_t).flatten().numpy()
            f2 = self.feature_extractor(ref_t).flatten().numpy()

        deep_diff = f1 - f2
        vec = np.concatenate([diff_stats, deep_diff], axis=0)
        return vec[:128]

    def predict(self, frame, detections):
        features = []

        for det in detections:
            if det['confidence'] < 0.7 or det['class'] not in self.reference_images:
                continue
            x1, y1, x2, y2 = map(int, det['bbox'])
            crop = frame[y1:y2, x1:x2]
            ref_crop = self.reference_images[det['class']]
            features.append(self._feature_vector(crop, ref_crop))

        if not features:
            features.append(np.zeros(128, dtype=np.float32))

        avg_feature = np.mean(features, axis=0)
        input_tensor = torch.tensor(avg_feature, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()[0]

        return prediction
