import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "ppe_dataset" / "images" / "train"
CSV_FILE = PROJECT_ROOT / "data" / "ppe_dataset" / "labels_classification.csv"
MODEL_PATH = PROJECT_ROOT / "ppe_model.pth"

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
VAL_SPLIT = 0.2
SEED = 42


class PPEDataset(Dataset):
    def __init__(self, rows, image_dir: Path, train: bool = True):
        self.rows = rows
        self.image_dir = image_dir

        if train:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = self.image_dir / row["image"]

        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        image = self.transform(image)

        label = torch.tensor(
            [
                float(row["helmet"]),
                float(row["vest"]),
                float(row["glasses"]),
            ],
            dtype=torch.float32
        )

        return image, label


def load_rows():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Fant ikke label-filen: {CSV_FILE}")

    rows = []
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"image", "helmet", "vest", "glasses"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV-filen må ha kolonnene: image, helmet, vest, glasses"
            )

        for row in reader:
            image_path = IMAGE_DIR / row["image"]
            if image_path.exists():
                rows.append(row)
            else:
                print(f"Advarsel: fant ikke bilde for rad: {row['image']}")

    if not rows:
        raise Exception("Ingen gyldige annoteringer funnet i CSV-filen.")

    return rows


def split_rows(rows, val_split=VAL_SPLIT):
    random.seed(SEED)
    rows = rows.copy()
    random.shuffle(rows)

    val_size = max(1, int(len(rows) * val_split))
    train_rows = rows[val_size:]
    val_rows = rows[:val_size]

    if len(train_rows) == 0:
        raise Exception("For få bilder til å lage train/val-splitt. Annoter flere bilder først.")

    return train_rows, val_rows


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0

    correct_helmet = 0
    correct_vest = 0
    correct_glasses = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            correct_helmet += (preds[:, 0] == labels[:, 0]).sum().item()
            correct_vest += (preds[:, 1] == labels[:, 1]).sum().item()
            correct_glasses += (preds[:, 2] == labels[:, 2]).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, len(loader))
    helmet_acc = correct_helmet / max(1, total)
    vest_acc = correct_vest / max(1, total)
    glasses_acc = correct_glasses / max(1, total)

    return avg_loss, helmet_acc, vest_acc, glasses_acc


def train():
    rows = load_rows()
    train_rows, val_rows = split_rows(rows)

    print(f"Totalt annoterte bilder: {len(rows)}")
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")

    train_dataset = PPEDataset(train_rows, IMAGE_DIR, train=True)
    val_dataset = PPEDataset(val_rows, IMAGE_DIR, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bruker device: {device}")

    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_loss, helmet_acc, vest_acc, glasses_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"helmet_acc={helmet_acc:.2%} | "
            f"vest_acc={vest_acc:.2%} | "
            f"glasses_acc={glasses_acc:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Ny beste modell lagret til: {MODEL_PATH}")

    print("\nTrening ferdig.")


if __name__ == "__main__":
    train()