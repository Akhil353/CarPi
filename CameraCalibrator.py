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
    def __init__(self, input_dim=64, output_dim=30):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
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

        resnet = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor()
        ])

        self.reference_images = self._load_reference_samples()

    def _load_reference_samples(self):
        refs = {}
        for cls in [0, 2, 16]:
            folder = os.path.join("calibrator_refs", str(cls))
            if not os.path.exists(folder):
                os.makedirs(folder)
            refs[cls] = [cv2.imread(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".jpg")]
        return refs

    def _blur_score(self, gray):
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        tenengrad = np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1)**2)
        fft_blur = np.mean(np.abs(np.fft.fft2(gray)))
        return lap_var, tenengrad, fft_blur

    def _smudge_score(self, img):
        flat = img.reshape(-1, 3)
        variance = np.var(flat, axis=0)
        return np.mean(variance < 15)

    def _corner_density(self, gray):
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        return len(corners) if corners is not None else 0

    def _compression_artifacts(self, gray):
        dct = cv2.dct(np.float32(gray[:32, :32]) / 255.0)
        return np.mean(np.abs(dct))

    def _rgb_to_hex(self, rgb):
        r, g, b = [int(c) for c in rgb]
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def _visual_stats(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_mean = img.mean(axis=(0, 1))
        rgb_std = img.std(axis=(0, 1))
        brightness = np.mean(gray)
        contrast = gray.std()
        lap_var, tenengrad, fft_blur = self._blur_score(gray)
        entropy = -np.sum((p := np.histogram(gray, bins=256)[0]/(gray.size + 1e-8)) * np.log2(p + 1e-8))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edge_mag = np.sqrt(sobelx**2 + sobely**2).mean()
        gamma = np.mean(np.power(img / 255.0 + 1e-8, 2.2))
        skew = ((gray - gray.mean())**3).mean() / (gray.std()**3 + 1e-8)
        kurtosis = ((gray - gray.mean())**4).mean() / (gray.std()**4 + 1e-8)
        smudge = self._smudge_score(img)
        corner = self._corner_density(gray)
        artifact = self._compression_artifacts(gray)
        return list(rgb_mean) + list(rgb_std) + [
            brightness, contrast, lap_var, tenengrad, fft_blur, entropy,
            edge_mag, gamma, skew, kurtosis, smudge, corner, artifact
        ], rgb_mean

    def _feature_vector(self, crop, ref_crop):
        if crop.size == 0 or ref_crop.size == 0:
            return np.zeros(64, dtype=np.float32), np.zeros(3, dtype=np.float32)

        stats_crop, rgb_crop = self._visual_stats(crop)
        stats_ref, _ = self._visual_stats(ref_crop)
        diff_stats = np.array(stats_crop) - np.array(stats_ref)

        crop_t = self.transform(crop).unsqueeze(0)
        ref_t = self.transform(ref_crop).unsqueeze(0)
        with torch.no_grad():
            f1 = self.feature_extractor(crop_t).flatten().numpy()
            f2 = self.feature_extractor(ref_t).flatten().numpy()

        deep_diff = f1 - f2
        vec = np.concatenate([diff_stats, deep_diff], axis=0)
        return vec[:64], rgb_crop

    def _analyze_grid_blindspots(self, frame, grid_size=6):
        h, w, _ = frame.shape
        gh, gw = h // grid_size, w // grid_size
        blindspots = []

        overall_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overall_focus = cv2.Laplacian(overall_gray, cv2.CV_64F).var()
        if overall_focus < 5.0:
            return []  # Skip blindspot analysis if entire image is already blurry

        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * gh, (i + 1) * gh
                x1, x2 = j * gw, (j + 1) * gw
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if lap_var < 6.0:
                    blindspots.append((i, j))
        return blindspots

    def predict(self, frame, detections):
        features = []
        tints = []

        for det in detections:
            if det['confidence'] < 0.7 or det['class'] not in self.reference_images:
                continue
            x1, y1, x2, y2 = map(int, det['bbox'])
            crop = frame[y1:y2, x1:x2]
            ref_crop = random.choice(self.reference_images[det['class']])
            vec, rgb = self._feature_vector(crop, ref_crop)
            features.append(vec * det['confidence'])
            tints.append(rgb)

        if not features:
            features.append(np.zeros(64, dtype=np.float32))
            tints.append(np.array([0, 0, 0], dtype=np.float32))

        avg_feature = np.mean(features, axis=0)
        avg_rgb = np.mean(tints, axis=0)
        input_tensor = torch.tensor(avg_feature, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()[0]

        blindspots = self._analyze_grid_blindspots(frame, grid_size=6)

        return {
            "prediction": prediction,
            "tint_hex": self._rgb_to_hex(avg_rgb),
            "avg_rgb": avg_rgb.tolist(),
            "blind_grid_coords": blindspots
        }
