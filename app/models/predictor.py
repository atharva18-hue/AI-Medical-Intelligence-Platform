"""
Prediction logic - preprocess image and run inference
"""

import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from app.models.cnn_model import CLASS_LABELS, load_trained_model

# standard imagenet normalization - using same as training
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ChestXrayPredictor:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            # use gpu if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = load_trained_model(model_path, self.device)
        self.labels = CLASS_LABELS

    def preprocess(self, image_bytes: bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0)  # add batch dim
        return tensor.to(self.device), img

    def predict(self, image_bytes: bytes):
        tensor, original_img = self.preprocess(image_bytes)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        result = {
            "predicted_class": self.labels[pred_idx.item()],
            "confidence": round(conf.item() * 100, 2),
            "probabilities": {
                self.labels[i]: round(probs[i].item() * 100, 2)
                for i in range(len(self.labels))
            },
        }
        return result, original_img
