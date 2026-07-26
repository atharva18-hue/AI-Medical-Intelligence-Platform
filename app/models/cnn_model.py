"""
CNN model for chest X-ray classification.
Used ResNet18 as backbone - learned this from pytorch tutorials.
"""

import torch
import torch.nn as nn
from torchvision import models


# disease classes we are detecting
CLASS_LABELS = ["Normal", "Pneumonia"]
NUM_CLASSES = len(CLASS_LABELS)


def build_model(pretrained=False):
    """Build ResNet18 based classifier"""
    # using resnet18 because its lighter and runs on my laptop gpu
    model = models.resnet18(weights="DEFAULT" if pretrained else None)

    # replace last fc layer for our 2 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model


def load_trained_model(weights_path, device="cpu"):
    model = build_model(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)

    # handle both formats - full checkpoint or just state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
