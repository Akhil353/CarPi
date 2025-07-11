import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from datetime import datetime
from torchvision.models import resnet18, ResNet18_Weights
from collections import deque
from pathlib import Path
import numpy as np

class CameraEffectsModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.features = list(base.children())
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        return self.classifier(feat)

    def extract_features(self, x):
        with torch.no_grad():
            return self.backbone(x).view(x.size(0), -1)

    def extract_intermediate(self, x, layer='layer1'):
        """
        Extract intermediate feature map from a given ResNet block (e.g., 'layer1').
        """
        outputs = {}

        def hook(module, input, output):
            outputs['feat'] = output.detach()

        # Pick ResNet block
        layer_map = {
            'layer1': 4,
            'layer2': 5,
            'layer3': 6,
            'layer4': 7,
        }
        hook_handle = self.features[layer_map[layer]].register_forward_hook(hook)
        _ = self.backbone(x)
        hook_handle.remove()
        return outputs['feat']  # Shape: [B, C, H, W]

class DriftDetector:
    def __init__(self, baseline_features, threshold=2.0, window_size=200):
        self.baseline = baseline_features
        self.threshold = threshold
        self.history = deque(maxlen=window_size)

    def detect(self, features):
        distance = torch.norm(features - self.baseline, dim=1).item()
        self.history.append(distance)
        is_drift = distance > self.threshold
        return is_drift, distance

    def should_log_drift(self, ratio=0.3):
        if len(self.history) < self.history.maxlen:
            return False
        return sum(d > self.threshold for d in self.history) / len(self.history) > ratio

class DriftLogger:
    def __init__(self, log_dir="drift_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, image, distance):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"drift_{timestamp}_d{distance:.2f}.jpg"
        cv2.imwrite(str(filename), image)
        print(f"⚠️ Logged drift image to {filename}")

class CameraMonitor:
    def __init__(self, model_path="camera_effects.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CameraEffectsModel(num_classes=len(checkpoint["idx2label"]))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Placeholder baseline vector (to be replaced with real feature mean)
        self.baseline = torch.zeros((1, 512)).to(self.device)
        self.drift_detector = DriftDetector(self.baseline)
        self.logger = DriftLogger()

    def update_baseline(self, baseline_features):
        self.drift_detector.baseline = baseline_features

    def process_frame(self, frame):
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        features = self.model.extract_features(img_tensor)
        is_drift, distance = self.drift_detector.detect(features)

        if is_drift and self.drift_detector.should_log_drift():
            self.logger.log_event(frame, distance)

        return is_drift, distance

    def get_blindspot_map(self, frame, layer='layer1'):
        """
        Returns a heatmap overlay showing regions of feature-level abnormality.
        """
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        feature_map = self.model.extract_intermediate(img_tensor, layer=layer)  # [1, C, H, W]

        activation_map = feature_map.norm(dim=1).squeeze(0).cpu().numpy()
        activation_map -= activation_map.min()
        activation_map /= (activation_map.max() + 1e-5)
        activation_map = cv2.resize(activation_map, (frame.shape[1], frame.shape[0]))

        heatmap = np.uint8(255 * activation_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        return overlay
