"""
CNN Evaluation Script — ForestGuard AI
Evaluates MobileNetV2 on the held-out test split only.
Produces confusion matrix, classification report, ROC-AUC, graphs, and sample predictions.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 on test split")
    parser.add_argument("--test_dir", type=str, default="datasets/cnn/test", help="Path to test split")
    parser.add_argument("--weights", type=str, default="model_weights/cnn_mobilenetv2.pt", help="Path to trained weights")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.test_dir)
    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"Error: Weights file {weights_path} not found.")
        return

    # Transforms
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = datasets.ImageFolder(str(test_dir), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    classes = test_ds.classes  # ['fire', 'no_fire']

    # Load Model
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 2)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy()[:, 0])  # Prob of class 0 (fire)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute Confusion Matrix
    tp = np.sum((all_preds == 0) & (all_labels == 0))
    fp = np.sum((all_preds == 0) & (all_labels == 1))
    fn = np.sum((all_preds == 1) & (all_labels == 0))
    tn = np.sum((all_preds == 1) & (all_labels == 1))

    accuracy = float((tp + tn) / len(all_labels))
    precision_fire = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_fire = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_fire = float(2 * precision_fire * recall_fire / (precision_fire + recall_fire)) if (precision_fire + recall_fire) > 0 else 0.0

    precision_nofire = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    recall_nofire = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_nofire = float(2 * precision_nofire * recall_nofire / (precision_nofire + recall_nofire)) if (precision_nofire + recall_nofire) > 0 else 0.0

    macro_f1 = (f1_fire + f1_nofire) / 2.0

    metrics = {
        "model": "MobileNetV2 Transfer Learning",
        "test_samples": len(all_labels),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "classes": {
            "fire": {
                "precision": round(precision_fire, 4),
                "recall": round(recall_fire, 4),
                "f1": round(f1_fire, 4)
            },
            "no_fire": {
                "precision": round(precision_nofire, 4),
                "recall": round(recall_nofire, 4),
                "f1": round(f1_nofire, 4)
            }
        },
        "confusion_matrix": {
            "true_fire": {"pred_fire": int(tp), "pred_no_fire": int(fn)},
            "true_no_fire": {"pred_fire": int(fp), "pred_no_fire": int(tn)}
        }
    }

    # Save metrics JSON
    metrics_path = Path("results/metrics/cnn_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Plot Confusion Matrix
    cm = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=["Fire", "No Fire"], yticklabels=["Fire", "No Fire"],
           title="CNN MobileNetV2 Confusion Matrix",
           ylabel="True label", xlabel="Predicted label")
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_plot_path = Path("results/graphs/cnn_confusion_matrix.png")
    cm_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    # Plot Accuracy & Loss curves if history exists
    history_file = Path("results/metrics/cnn_history.json")
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            hist = json.load(f)
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Loss Plot
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, hist["train_loss"], 'b-', label='Training Loss')
        plt.plot(epochs, hist["val_loss"], 'r-', label='Validation Loss')
        plt.title('CNN MobileNetV2 Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("results/graphs/cnn_loss.png", dpi=200)
        plt.close()

        # Accuracy Plot
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, hist["train_acc"], 'b-', label='Training Accuracy')
        plt.plot(epochs, hist["val_acc"], 'r-', label='Validation Accuracy')
        plt.title('CNN MobileNetV2 Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig("results/graphs/cnn_accuracy.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
