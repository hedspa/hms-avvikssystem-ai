import csv
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "ppe_dataset" / "images" / "train"
CSV_FILE = PROJECT_ROOT / "data" / "ppe_dataset" / "labels_classification.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def load_images():
    images = [
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix in VALID_EXTENSIONS
    ]
    return sorted(images)


def ensure_csv_exists():
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "helmet", "vest", "glasses"])


def load_existing_annotations():
    rows = {}

    if CSV_FILE.exists():
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows[row["image"]] = row

    return rows


def ask_binary(prompt: str) -> int:
    while True:
        answer = input(prompt).strip()
        if answer in {"0", "1"}:
            return int(answer)
        print("Skriv bare 1 for ja eller 0 for nei.")


def save_all_rows(rows_dict):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "helmet", "vest", "glasses"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_name in sorted(rows_dict.keys()):
            writer.writerow(rows_dict[image_name])


def show_image(img_path: Path, index: int, total: int):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Kunne ikke lese bilde: {img_path}")

    h, w = img.shape[:2]
    max_w, max_h = 1200, 800
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.putText(
        img,
        f"{img_path.name} ({index}/{total})",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Annotate", img)
    cv2.waitKey(1)


def needs_glasses_annotation(row: dict) -> bool:
    return "glasses" not in row or row["glasses"] in {"", None}


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")

    ensure_csv_exists()

    images = load_images()
    if not images:
        raise Exception(f"Ingen bilder funnet i: {IMAGE_DIR}")

    rows_dict = load_existing_annotations()

    remaining_images = []
    for img in images:
        if img.name not in rows_dict:
            remaining_images.append(img)
        else:
            if needs_glasses_annotation(rows_dict[img.name]):
                remaining_images.append(img)

    if not remaining_images:
        print("Alle bilder er allerede annotert for helmet, vest og glasses.")
        return

    print(f"Fant {len(images)} bilder totalt.")
    print(f"{len(remaining_images)} bilder mangler glasses eller full annotering.\n")

    for idx, img_path in enumerate(remaining_images, start=1):
        show_image(img_path, idx, len(remaining_images))

        existing = rows_dict.get(img_path.name, {})
        old_helmet = existing.get("helmet")
        old_vest = existing.get("vest")
        old_glasses = existing.get("glasses")

        print(f"\nBilde: {img_path.name} ({idx}/{len(remaining_images)})")
        if old_helmet is not None:
            print(f"Eksisterende helmet: {old_helmet}")
        if old_vest is not None:
            print(f"Eksisterende vest: {old_vest}")
        if old_glasses not in {None, ''}:
            print(f"Eksisterende glasses: {old_glasses}")

        if old_helmet in {"0", "1"} and old_vest in {"0", "1"}:
            print("Beholder eksisterende helmet og vest.")
            helmet = int(old_helmet)
            vest = int(old_vest)
        else:
            helmet = ask_binary("Hjelm? (1/0): ")
            vest = ask_binary("Vest? (1/0): ")

        glasses = ask_binary("Briller? (1/0): ")

        rows_dict[img_path.name] = {
            "image": img_path.name,
            "helmet": helmet,
            "vest": vest,
            "glasses": glasses,
        }

        save_all_rows(rows_dict)
        print(
            f"Lagret: {img_path.name} -> helmet={helmet}, vest={vest}, glasses={glasses}"
        )

        cv2.destroyWindow("Annotate")
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print("\nAnnotering ferdig.")


if __name__ == "__main__":
    main()