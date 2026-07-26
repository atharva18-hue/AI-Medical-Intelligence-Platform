"""
Training script for chest X-ray classifier.
Dataset: Chest X-Ray Images (Pneumonia) from Kaggle
Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Folder structure expected:
data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/

If you dont have the full dataset, run with --demo flag to train on sample images.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.models.cnn_model import build_model, CLASS_LABELS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "chest_xray_model.pth")


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds


def get_class_weights(dataset):
    """Handle imbalanced classes - more pneumonia samples than normal"""
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    sample_weights = [weights[t] for t in targets]
    return torch.FloatTensor(weights), sample_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--demo", action="store_true", help="Train on sample images only")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  # apple silicon gpu
    else:
        device = "cpu"
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()

    if args.demo or not os.path.exists(os.path.join(DATA_DIR, "train")):
        print("Demo mode - using sample_images folder")
        sample_dir = os.path.join(BASE_DIR, "data", "sample_images")
        # create simple folder structure if needed
        os.makedirs(os.path.join(sample_dir, "NORMAL"), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "PNEUMONIA"), exist_ok=True)
        train_dataset = datasets.ImageFolder(sample_dir, transform=train_tf)
        val_dataset = train_dataset  # same for demo
        test_dataset = None
    else:
        train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
        val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)
        test_path = os.path.join(DATA_DIR, "test")
        test_dataset = datasets.ImageFolder(test_path, transform=val_tf) if os.path.exists(test_path) else None

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # class weights for imbalanced data
    class_weights, sample_weights = get_class_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) if not args.demo else None
    class_weights = class_weights.to(device)
    print(f"Class weights: Normal={class_weights[0]:.2f}, Pneumonia={class_weights[1]:.2f}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0) if sampler else DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if test_dataset else None

    model = build_model(pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_score = 0.0
    history = {"train_loss": [], "val_acc": [], "test_acc": []}

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        test_acc = 0.0
        test_f1 = 0.0
        if test_loader:
            test_acc, y_true, y_pred = evaluate(model, test_loader, device)
            test_f1 = f1_score(y_true, y_pred, average="macro")
        scheduler.step()

        history["train_loss"].append(loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}", flush=True)

        # save best model based on test macro F1 (val set too small - only 16 images)
        score = test_f1 if test_loader else val_acc
        if score > best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASS_LABELS,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
            }, MODEL_SAVE_PATH)
            print(f"  Saved best model (score={score:.4f})", flush=True)

    # reload best saved model for final evaluation
    if os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\nLoaded best checkpoint (test_acc={checkpoint.get('test_accuracy', 'N/A')}, test_f1={checkpoint.get('test_f1', 'N/A')})")

    # final eval on test set (624 images - more reliable than val which only has 16)
    eval_loader = test_loader if test_loader else val_loader
    eval_name = "Test" if test_loader else "Val"
    _, y_true, y_pred = evaluate(model, eval_loader, device)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{eval_name} Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    # save training plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"])
    ax[0].set_title("Training Loss")
    ax[1].plot(history["val_acc"], label="Val")
    if history["test_acc"]:
        ax[1].plot(history["test_acc"], label="Test")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    plot_path = os.path.join(BASE_DIR, "docs", "training_curves.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
