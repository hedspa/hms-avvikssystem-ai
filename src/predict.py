from pathlib import Path

from ultralytics import YOLO

from src.config import DEFAULT_CONFIDENCE, DEFAULT_IMAGE_SIZE, DEFAULT_IOU, OUTPUT_DIR
from src.rules import Detection, draw_results, evaluate_detections
from src.utils import build_output_paths, load_image, save_image, save_json


def yolo_results_to_detections(result) -> list[Detection]:
    names = result.names
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(
                class_name=class_name,
                confidence=float(box.conf.item()),
                box=(int(x1), int(y1), int(x2), int(y2)),
            )
        )
    return detections


def sanitize_results(results: list[dict]) -> list[dict]:
    sanitized = []
    for result in results:
        sanitized.append(
            {
                "person_id": result["person_id"],
                "helmet": result["helmet"],
                "vest": result["vest"],
                "deviation": result["deviation"],
            }
        )
    return sanitized


def predict_image(
    image_path: str | Path,
    weights_path: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMAGE_SIZE,
) -> dict:
    model = YOLO(str(weights_path))
    image = load_image(image_path)
    prediction = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    detections = yolo_results_to_detections(prediction)
    evaluated = evaluate_detections(detections)
    output = sanitize_results(evaluated)

    annotated = draw_results(image.copy(), evaluated)
    json_path, annotated_path = build_output_paths(image_path, output_dir)
    save_json(output, json_path)
    save_image(annotated, annotated_path)

    return {
        "results": output,
        "json_path": json_path,
        "annotated_image_path": annotated_path,
    }
