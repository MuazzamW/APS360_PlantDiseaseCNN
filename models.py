import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from torchvision import models
from collections import Counter

num_classes = 27
IMG_SIZE = 224

#primaryy and baseline classes
class BaselineANN(nn.Module):
    def __init__(self, num_classes=27, img_size=IMG_SIZE):
        super(BaselineANN, self).__init__()

        self.name = "baseline_ann"

        input_size = 3 * img_size * img_size

        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]

        x = x.view(x.size(0), -1)
        # x shape: [batch_size, 150528]

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        # x shape: [batch_size, num_classes]

        return x
    
class PrimaryCNN(nn.Module):
    def __init__(self, num_classes=27, img_size=224):
        super(PrimaryCNN, self).__init__()

        self.name = "primary_cnn"

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        flattened_size = self._get_flattened_size(img_size)

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.config = {
            "num_classes": num_classes,
            "img_size": img_size,
            "conv_filters": [32, 64, 128],
            "kernel_size": 3,
            "padding": 1,
            "dropout": 0.3,
            "fc_hidden": 256
        }

    def _get_flattened_size(self, img_size):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)

            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            x = x.view(x.size(0), -1)

            return x.shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x
    

class TransferResNet18(nn.Module):
    def __init__(self, num_classes=27, freeze_backbone=True):
        super(TransferResNet18, self).__init__()

        self.name = "transfer_resnet18"

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(num_features, num_classes)

        self.config = {
            "num_classes": num_classes,
            "backbone": "resnet18",
            "pretrained": True,
            "freeze_backbone": freeze_backbone,
            "final_layer": f"Linear({num_features}, {num_classes})"
        }

    def forward(self, x):
        return self.model(x)


#utilities

def get_class_weights(train_dataset):
    labels = [label for _, label in train_dataset.samples]

    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)

    weights = []

    for class_idx in range(num_classes):
        class_count = class_counts[class_idx]
        weight = total_samples / (num_classes * class_count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)

def plot_training_curve(
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
    save_dir=None,
    show_plots=True
):
    num_epochs = len(train_loss)
    epochs = range(1, num_epochs + 1)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)


    plt.figure()

    plt.plot(
        epochs,
        train_accuracy,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy"
    )

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    if save_dir is not None:
        plt.savefig(
            save_dir / "accuracy_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

    if show_plots:
        plt.show()
    else:
        plt.close()

    plt.figure()

    plt.plot(
        epochs,
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss"
    )

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    if save_dir is not None:
        plt.savefig(
            save_dir / "loss_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

    if show_plots:
        plt.show()
    else:
        plt.close()

def create_run_directory(
    model,
    num_epochs,
    learning_rate,
    batch_size=None,
    results_root="results"
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = getattr(model, "name", model.__class__.__name__)

    run_name = (
        f"{model_name}_"
        f"{timestamp}_"
        f"epochs{num_epochs}_"
        f"lr{learning_rate}"
    )

    if batch_size is not None:
        run_name += f"_batch{batch_size}"

    run_dir = Path(results_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir

def save_config(
    run_dir,
    model,
    num_epochs,
    learning_rate,
    optimizer_name,
    criterion_name,
    train_loader,
    val_loader,
    extra_config=None
):
    config = {
        "model_name": getattr(model, "name", model.__class__.__name__),
        "model_class": model.__class__.__name__,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "criterion": criterion_name,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "batch_size": train_loader.batch_size,
        "device_available": "cuda" if torch.cuda.is_available() else "cpu",
        "img_size": IMG_SIZE,
        "num_classes": num_classes
    }

    if extra_config is not None:
        config.update(extra_config)
    
    if hasattr(model, "config"):
        config["model_config"] = model.config

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def save_metrics_csv(
    run_dir,
    train_loss_array,
    train_accuracy_array,
    val_loss_array,
    val_accuracy_array
):
    num_epochs = len(train_loss_array)

    metrics_df = pd.DataFrame({
        "epoch": np.arange(1, num_epochs + 1),
        "train_loss": train_loss_array,
        "train_accuracy": train_accuracy_array,
        "val_loss": val_loss_array,
        "val_accuracy": val_accuracy_array
    })

    metrics_df.to_csv(
        run_dir / "metrics.csv",
        index=False
    )

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate=0.001,
    results_root="results",
    save_results=True,
    show_plots=True
):
    train_loss_array = np.zeros(num_epochs)
    train_accuracy_array = np.zeros(num_epochs)

    val_loss_array = np.zeros(num_epochs)
    val_accuracy_array = np.zeros(num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    class_weights = get_class_weights(train_loader.dataset).to(device)

    #weighted criterion so that mistakes on underrepresented classes count more
    #tries to offset the imbalance created by having more plant village images than plant doc
    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    run_dir = None

    if save_results:
        run_dir = create_run_directory(
            model=model,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=train_loader.batch_size,
            results_root=results_root
        )

        save_config(
            run_dir=run_dir,
            model=model,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_name="Adam",
            criterion_name="CrossEntropyLoss",
            train_loader=train_loader,
            val_loader=val_loader
        )

        print(f"Saving results to: {run_dir}")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        train_loss_array[epoch] = train_loss
        train_accuracy_array[epoch] = train_acc

        val_loss_array[epoch] = val_loss
        val_accuracy_array[epoch] = val_acc

    if save_results:
        save_metrics_csv(
            run_dir=run_dir,
            train_loss_array=train_loss_array,
            train_accuracy_array=train_accuracy_array,
            val_loss_array=val_loss_array,
            val_accuracy_array=val_accuracy_array
        )

        final_metrics = {
            "final_train_loss": float(train_loss_array[-1]),
            "final_train_accuracy": float(train_accuracy_array[-1]),
            "final_val_loss": float(val_loss_array[-1]),
            "final_val_accuracy": float(val_accuracy_array[-1]),
            "best_val_accuracy": float(np.max(val_accuracy_array)),
            "best_val_accuracy_epoch": int(np.argmax(val_accuracy_array) + 1),
            "lowest_val_loss": float(np.min(val_loss_array)),
            "lowest_val_loss_epoch": int(np.argmin(val_loss_array) + 1)
        }

        with open(run_dir / "final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=4)

        torch.save(
            model.state_dict(),
            run_dir / "model_state_dict.pt"
        )

    plot_training_curve(
        train_loss_array,
        train_accuracy_array,
        val_loss_array,
        val_accuracy_array,
        save_dir=run_dir if save_results else None,
        show_plots=show_plots
    )

    return (
        model,
        train_loss_array,
        train_accuracy_array,
        val_loss_array,
        val_accuracy_array,
        run_dir
    )


def test(model, test_loader, save_results, run_dir):
    test_loss = None
    test_acc = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if test_loader is not None:
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(
            f"\nFinal Test Loss: {test_loss:.4f}, "
            f"Final Test Acc: {test_acc:.4f}"
        )

        if save_results:
            test_metrics = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc)
            }

            with open(run_dir / "test_metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=4)
    
        