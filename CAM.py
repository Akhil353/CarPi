import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import cv2
from pathlib import Path
import numpy as np
import time
import json
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# MODEL: CameraEffectsModel
# ─────────────────────────────────────────────────────────────

class CameraEffectsModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        return self.classifier(feat)

    def extract_features(self, x):
        return self.backbone(x).view(x.size(0), -1)

    def extract_intermediate(self, x, layer='layer1'):
        modules = dict(self.backbone.named_children())
        out = x
        for name, block in modules.items():
            out = block(out)
            if name == layer:
                return out
        return out

# ─────────────────────────────────────────────────────────────
# BLUR DETECTOR
# ─────────────────────────────────────────────────────────────

class BlurDetector:
    def __init__(self, blur_thresh=50, brightness_thresh=50, contrast_thresh=15, edge_thresh=0.02, min_area=500):
        self.blur_thresh = blur_thresh
        self.brightness_thresh = brightness_thresh
        self.contrast_thresh = contrast_thresh
        self.edge_thresh = edge_thresh
        self.min_area = min_area

    def analyze_region(self, gray, region_mask):
        region = gray[region_mask]
        if region.size == 0:
            return None, None

        # Metrics
        lap_var = cv2.Laplacian(region, cv2.CV_64F).var()
        brightness = np.mean(region)
        contrast = np.std(region)
        edges = cv2.Canny(region, 100, 200)
        edge_density = np.sum(edges > 0) / (region.size + 1e-5)

        # ────── NEW: Only trigger if *all* are bad ──────
        is_blurry = lap_var < self.blur_thresh
        is_low_contrast = contrast < self.contrast_thresh
        is_low_edges = edge_density < self.edge_thresh
        is_very_dark = brightness < self.brightness_thresh

        cause = None

        if is_blurry and (is_low_contrast or is_low_edges):
            if is_very_dark:
                cause = "low_light"
            elif is_low_edges:
                cause = "low_edges"
            elif is_low_contrast:
                cause = "low_contrast"
            else:
                cause = "flat"

        return lap_var, cause




    def detect_blindspots(self, frame, model=None, transform=None, threshold_blur=60):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        boxes = []
        step_x = w // 8
        step_y = h // 6

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                x_end = min(x + step_x, w)
                y_end = min(y + step_y, h)
                region = gray[y:y_end, x:x_end]
                color_region = frame[y:y_end, x:x_end]

                # 1. Calculate Laplacian variance
                lap_var = cv2.Laplacian(region, cv2.CV_64F).var()
                brightness = np.mean(region)
                contrast = np.std(region)
                edges = cv2.Canny(region, 100, 200)
                edge_density = np.sum(edges > 0) / (region.size + 1e-5)

                # 2. Check model confusion (optional)
                model_confused = False
                if model and transform:
                    try:
                        patch_tensor = transform(color_region).unsqueeze(0).to(next(model.parameters()).device)
                        with torch.no_grad():
                            out = model(patch_tensor)
                            probs = torch.softmax(out, dim=1)
                            max_conf = probs.max().item()
                        model_confused = max_conf < 0.5
                    except:
                        pass

                # 3. Combine all criteria
                blurry = lap_var < threshold_blur and edge_density < self.edge_thresh and contrast < self.contrast_thresh

                if blurry and model_confused:
                    boxes.append({
                        "bbox": (x, y, x_end, y_end),
                        "cause": f"blurry+confused",
                        "lap_var": round(lap_var, 2),
                        "contrast": round(contrast, 2),
                        "edge_density": round(edge_density, 4)
                    })

        return boxes


# ─────────────────────────────────────────────────────────────
# DRIFT LOGGER
# ─────────────────────────────────────────────────────────────

class DriftLogger:
    def __init__(self, log_dir="drift_logs", cooldown=5.0, max_history=6):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.last_log_time = 0
        self.history = deque(maxlen=max_history)

    def log(self, image, distance, blindspots, feature=None):
        now = time.time()
        if now - self.last_log_time < self.cooldown:
            return False

        # Compare with last image in history
        if self.history:
            last = self.history[-1]
            last_img = cv2.imread(last["img_path"])
            if last_img is not None:
                diff = np.mean(cv2.absdiff(last_img, image))
                if diff < 10:
                    print("⚠️ Skipped log: too visually similar to previous.")
                    return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"drift_{timestamp}_d{distance:.2f}"
        img_path = self.log_dir / f"{filename_base}.jpg"
        meta_path = self.log_dir / f"{filename_base}.json"

        cv2.imwrite(str(img_path), image)

        metadata = {
            "timestamp": timestamp,
            "distance": distance,
            "blindspots": blindspots
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.history.append({
            "img_path": str(img_path),
            "meta_path": str(meta_path),
            "features": feature.cpu() if feature is not None else None
        })

        self.last_log_time = now
        print(f"⚠️ Logged drift to {img_path.name} with {len(blindspots)} blindspots.")
        return True


    def get_recent_logs(self):
        return list(self.history)

# ─────────────────────────────────────────────────────────────
# BASELINE UPDATER
# ─────────────────────────────────────────────────────────────

class BaselineUpdater:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def update_baseline_from_logs(self, logger):
        logs = logger.get_recent_logs()
        if len(logs) < 6:
            print(f"ℹ️ Not enough logs to update baseline ({len(logs)}/6)")
            return None

        features_list = []
        for entry in logs:
            img = cv2.imread(entry["img_path"])
            if img is None:
                continue
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.extract_features(input_tensor)
                features_list.append(feat)

        if not features_list:
            print("⚠️ No valid features extracted.")
            return None

        new_baseline = torch.mean(torch.cat(features_list, dim=0), dim=0, keepdim=True)
        print("✅ Updated baseline from recent drift logs.")
        return new_baseline

# ─────────────────────────────────────────────────────────────
# DRIFT DETECTOR
# ─────────────────────────────────────────────────────────────

class DriftDetector:
    def __init__(self, baseline, threshold=3.0):
        self.baseline = baseline
        self.threshold = threshold

    def detect(self, current_features):
        distance = torch.norm(current_features - self.baseline, dim=1).item()
        is_drift = distance > self.threshold
        return is_drift, distance

    def set_baseline(self, new_baseline):
        self.baseline = new_baseline

# ─────────────────────────────────────────────────────────────
# CAMERA MONITOR (MAIN CLASS)
# ─────────────────────────────────────────────────────────────

class CameraMonitor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        checkpoint = torch.load("camera_effects.pth", map_location=device)
        num_classes = len(checkpoint["idx2label"])
        self.model = CameraEffectsModel(num_classes).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.blur_detector = BlurDetector()
        self.logger = DriftLogger()
        self.updater = BaselineUpdater(self.model, device)
        self.baseline = torch.zeros((1, 512)).to(device)
        self.drift_detector = DriftDetector(self.baseline)
        self.last_retrain = time.time()
        self.retrain_interval = 30

    def process_frame(self, frame):
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.extract_features(img_tensor)

        is_drift, distance = self.drift_detector.detect(features)
        blindspots = self.blur_detector.detect_blindspots(frame, model=self.model, transform=self.transform)


        if is_drift and len(blindspots) >= 5:
            logged = self.logger.log(frame, distance, blindspots, feature=features)
            
            logs = self.logger.get_recent_logs()
            if logged and len(logs) >= 6:
                # Check if logs are diverse enough (feature variance)
                all_feats = [entry["features"] for entry in logs if entry["features"] is not None]
                if len(all_feats) >= 6:
                    stacked = torch.cat(all_feats, dim=0)
                    diversity = torch.var(stacked, dim=0).mean().item()
                    print(f"🧠 Drift log diversity: {diversity:.4f}")
                    
                    if diversity > 0.01:  # heuristic threshold
                        new_baseline = self.updater.update_baseline_from_logs(self.logger)
                        if new_baseline is not None:
                            self.drift_detector.set_baseline(new_baseline)
                            self.retrain_model_from_logs()

            
        return is_drift, distance, blindspots
    
    def retrain_model_from_logs(self, epochs=3, lr=1e-5):
        logs = self.logger.get_recent_logs()
        if len(logs) < 6:
            return

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            count = 0

            for entry in logs:
                img = cv2.imread(entry["img_path"])
                if img is None:
                    continue

                input_tensor = self.transform(img).unsqueeze(0).to(self.device)

                pred_feat = self.model.extract_features(input_tensor)
                target_feat = self.baseline.to(self.device)

                loss = loss_fn(pred_feat, target_feat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / (count or 1)
            print(f"🔁 Retrain Epoch {epoch+1}: Loss = {avg_loss:.6f}")

        self.model.eval()
