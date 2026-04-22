from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "ppe_model.pth"
TEST_DIR = PROJECT_ROOT / "data" / "ppe_dataset" / "images" / "test"

IMAGE_SIZE = 224
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])


def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(model, image_path):
    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.sigmoid(output)

    helmet_prob = probs[0][0].item()
    vest_prob = probs[0][1].item()
    glasses_prob = probs[0][2].item()

    helmet_pred = 1 if helmet_prob > 0.5 else 0
    vest_pred = 1 if vest_prob > 0.5 else 0
    glasses_pred = 1 if glasses_prob > 0.5 else 0

    return helmet_pred, vest_pred, glasses_pred, helmet_prob, vest_prob, glasses_prob


def main():
    model = load_model()

    images = sorted([
        p for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ])

    print(f"\nTester {len(images)} bilder\n")

    for img_path in images:
        try:
            helmet, vest, glasses, hp, vp, gp = predict(model, img_path)

            helmet_txt = "YES" if helmet else "NO"
            vest_txt = "YES" if vest else "NO"
            glasses_txt = "YES" if glasses else "NO"

            print(f"{img_path.name}")
            print(f"  Helmet:  {helmet_txt} ({hp:.2f})")
            print(f"  Vest:    {vest_txt} ({vp:.2f})")
            print(f"  Glasses: {glasses_txt} ({gp:.2f})")
            print()

        except Exception as e:
            print(f"Hopper over {img_path.name}: {e}\n")


if __name__ == "__main__":
    main()