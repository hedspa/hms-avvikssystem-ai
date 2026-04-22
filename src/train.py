from pathlib import Path

from ultralytics import YOLO

from src.config import DATASET_CONFIG, DEFAULT_EPOCHS, DEFAULT_IMAGE_SIZE


def train_model(
    model_name: str = "yolov8n.pt",
    data_config: str | Path = DATASET_CONFIG,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    project: str = "runs/ppe",
    run_name: str = "train",
):
    model = YOLO(model_name)
    return model.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=run_name,
    )


if __name__ == "__main__":
    train_model()
