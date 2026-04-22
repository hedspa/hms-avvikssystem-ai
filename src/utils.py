import json
from pathlib import Path
from typing import Any

import cv2


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(image_path: str | Path):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def save_image(image, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Could not write image: {output_path}")
    return output_path


def save_json(data: Any, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def build_output_paths(image_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    image_path = Path(image_path)
    output_dir = ensure_dir(Path(output_dir))
    stem = image_path.stem
    json_path = output_dir / f"{stem}.json"
    annotated_path = output_dir / f"{stem}_annotated.jpg"
    return json_path, annotated_path


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2
