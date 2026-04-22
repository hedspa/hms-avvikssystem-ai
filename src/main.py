import argparse
from pathlib import Path

from src.config import DATASET_CONFIG, DEFAULT_EPOCHS, DEFAULT_IMAGE_SIZE, OUTPUT_DIR
from src.predict import predict_image
from src.train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP for PPE detection on construction site images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a YOLO model on the PPE dataset.")
    train_parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model or existing checkpoint.")
    train_parser.add_argument("--data", default=str(DATASET_CONFIG), help="Path to dataset YAML.")
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    train_parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Training image size.")
    train_parser.add_argument("--project", default="runs/ppe", help="Ultralytics project folder.")
    train_parser.add_argument("--name", default="train", help="Run name for Ultralytics.")

    predict_parser = subparsers.add_parser("predict", help="Run inference on one image and save outputs.")
    predict_parser.add_argument("--image", required=True, help="Input image path.")
    predict_parser.add_argument("--weights", required=True, help="Path to trained weights.")
    predict_parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for JSON and annotated image.")
    predict_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    predict_parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    predict_parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Inference image size.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_model(
            model_name=args.model,
            data_config=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            project=args.project,
            run_name=args.name,
        )
        return

    if args.command == "predict":
        result = predict_image(
            image_path=Path(args.image),
            weights_path=Path(args.weights),
            output_dir=Path(args.output_dir),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
        )
        print(result["json_path"])
        print(result["annotated_image_path"])
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
