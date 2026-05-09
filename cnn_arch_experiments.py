from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.0003
criterion = nn.CrossEntropyLoss()


class PadResize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)

        pad_w = (max_side - w) // 2
        pad_h = (max_side - h) // 2

        padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
        img = transforms.functional.pad(img, padding, fill=0)
        img = img.resize((self.size, self.size))
        return img


# Recreated exactly from notebook
train_transform_aug = transforms.Compose(
    [
        PadResize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

train_transform_noaug = transforms.Compose(
    [
        PadResize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

val_test_transform = transforms.Compose(
    [
        PadResize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Dataset paths (exact notebook strings)
seat_train_dir = "data/splits/Seat Cover/train"
seat_val_dir = "data/splits/Seat Cover/val"
seat_test_dir = "data/splits/Seat Cover/test"


def build_seat_dataloaders():
    for p in [seat_train_dir, seat_val_dir, seat_test_dir]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing dataset path: {p}")

    seat_train_noaug = datasets.ImageFolder(seat_train_dir, transform=train_transform_noaug)
    seat_train_aug = datasets.ImageFolder(seat_train_dir, transform=train_transform_aug)
    seat_val = datasets.ImageFolder(seat_val_dir, transform=val_test_transform)
    seat_test = datasets.ImageFolder(seat_test_dir, transform=val_test_transform)

    seat_train_loader_noaug = DataLoader(seat_train_noaug, batch_size=BATCH_SIZE, shuffle=True)
    seat_train_loader_aug = DataLoader(seat_train_aug, batch_size=BATCH_SIZE, shuffle=True)
    seat_val_loader = DataLoader(seat_val, batch_size=BATCH_SIZE, shuffle=False)
    seat_test_loader = DataLoader(seat_test, batch_size=BATCH_SIZE, shuffle=False)

    return {
        "seat_train_noaug": seat_train_noaug,
        "seat_train_aug": seat_train_aug,
        "seat_val": seat_val,
        "seat_test": seat_test,
        "seat_train_loader_noaug": seat_train_loader_noaug,
        "seat_train_loader_aug": seat_train_loader_aug,
        "seat_val_loader": seat_val_loader,
        "seat_test_loader": seat_test_loader,
    }


class ConvBNReLUPoolDrop(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p_drop: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p_drop),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLUPoolDrop(3, 32, 0.10),
            ConvBNReLUPoolDrop(32, 64, 0.15),
            ConvBNReLUPoolDrop(64, 128, 0.20),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class BNDropoutCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLUPoolDrop(3, 32, 0.15),
            ConvBNReLUPoolDrop(32, 64, 0.20),
            ConvBNReLUPoolDrop(64, 128, 0.25),
            ConvBNReLUPoolDrop(128, 128, 0.30),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class DeeperCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLUPoolDrop(3, 32, 0.10),
            ConvBNReLUPoolDrop(32, 64, 0.15),
            ConvBNReLUPoolDrop(64, 128, 0.20),
            ConvBNReLUPoolDrop(128, 256, 0.25),
            ConvBNReLUPoolDrop(256, 256, 0.30),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer, name: str) -> Dict[str, List[float]]:
    train_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train if total_train else 0.0

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val if total_val else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"{name} | Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
    }


def main() -> None:
    print(f"Using device: {device}")
    print(f"torchvision version: {torchvision.__version__}")

    data = build_seat_dataloaders()
    seat_train_loader_noaug = data["seat_train_loader_noaug"]
    seat_val_loader = data["seat_val_loader"]

    model_specs: List[Tuple[str, nn.Module]] = [
        ("BaselineCNN", BaselineCNN(num_classes=2)),
        ("BNDropoutCNN", BNDropoutCNN(num_classes=2)),
        ("DeeperCNN", DeeperCNN(num_classes=2)),
    ]

    results_summary: Dict[str, float] = {}

    for model_name, model in model_specs:
        print("\n" + "=" * 80)
        print(f"Training {model_name}")
        print("=" * 80)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        history = train_model(model, seat_train_loader_noaug, seat_val_loader, optimizer, model_name)
        best_val_acc = float(history["best_val_acc"])
        results_summary[model_name] = best_val_acc
        print(f"{model_name} best validation accuracy: {best_val_acc:.4f}")

    print("\nFinal Results (best validation accuracy)")
    for model_name, acc in results_summary.items():
        print(f"- {model_name}: {acc:.4f}")

    best_name, best_acc = max(results_summary.items(), key=lambda x: x[1])
    print(f"\nBest architecture: {best_name} ({best_acc:.4f})")


if __name__ == "__main__":
    main()
