"""
MobileNetV2 Transfer Learning for Forest Fire Detection — ForestGuard AI
Two-stage training: frozen backbone head warmup -> partial fine-tuning.
"""

import os
import time
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Fire Classification")
    parser.add_argument("--data_dir", type=str, default="datasets/cnn", help="Path to cnn dataset folder")
    parser.add_argument("--epochs_stage1", type=int, default=5, help="Head warmup epochs")
    parser.add_argument("--epochs_stage2", type=int, default=8, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--out_dir", type=str, default="model_weights", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),  # Tiny hue!
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=val_transform)

    print(f"Classes found: {train_ds.classes}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze backbone for Stage 1
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier: Dropout(0.2) -> Linear(1280, 2)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 2)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights_path = out_dir / "cnn_mobilenetv2.pt"

    print("=== Stage 1: Warmup Classifier Head ===")
    for epoch in range(args.epochs_stage1):
        t0 = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))

        print(f"Epoch {epoch+1}/{args.epochs_stage1} [{dt:.1f}s] - Train Loss: {t_loss:.4f}, Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f}, Acc: {v_acc:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), best_weights_path)

    print("=== Stage 2: Unfreeze Upper Layers for Fine-Tuning ===")
    # Unfreeze last blocks of MobileNetV2
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)

    for epoch in range(args.epochs_stage2):
        t0 = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))

        print(f"Epoch {epoch+1}/{args.epochs_stage2} [{dt:.1f}s] - Train Loss: {t_loss:.4f}, Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f}, Acc: {v_acc:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), best_weights_path)

    # Save training history
    history_path = Path("results/metrics/cnn_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training finished! Best model saved to {best_weights_path}")
    print(f"History logged to {history_path}")

if __name__ == "__main__":
    main()
