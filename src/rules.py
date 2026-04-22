from dataclasses import dataclass

import cv2

from src.config import (
    CLASS_HELMET,
    CLASS_PERSON,
    CLASS_VEST,
    HEAD_REGION_RATIO,
    TORSO_REGION_END_RATIO,
    TORSO_REGION_START_RATIO,
)
from src.utils import clip_box


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def split_person_regions(person_box: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    head_y2 = y1 + int(height * HEAD_REGION_RATIO)
    torso_y1 = y1 + int(height * TORSO_REGION_START_RATIO)
    torso_y2 = y1 + int(height * TORSO_REGION_END_RATIO)
    return {
        "head": (x1, y1, x2, head_y2),
        "torso": (x1, torso_y1, x2, torso_y2),
    }


def find_best_match(region_box: tuple[int, int, int, int], detections: list[Detection], class_name: str) -> Detection | None:
    candidates = []
    for detection in detections:
        if detection.class_name != class_name:
            continue
        if point_in_box(box_center(detection.box), region_box):
            candidates.append(detection)

    if not candidates:
        return None
    return max(candidates, key=lambda det: det.confidence)


def evaluate_person(person_detection: Detection, detections: list[Detection], person_id: int) -> dict:
    regions = split_person_regions(person_detection.box)
    helmet_detection = find_best_match(regions["head"], detections, CLASS_HELMET)
    vest_detection = find_best_match(regions["torso"], detections, CLASS_VEST)

    helmet = helmet_detection is not None
    vest = vest_detection is not None
    deviation = []
    if not helmet:
        deviation.append("no-helmet")
    if not vest:
        deviation.append("no-vest")

    return {
        "person_id": person_id,
        "helmet": helmet,
        "vest": vest,
        "deviation": deviation,
        "person_box": person_detection.box,
        "head_box": regions["head"],
        "torso_box": regions["torso"],
    }


def evaluate_detections(detections: list[Detection]) -> list[dict]:
    people = [det for det in detections if det.class_name == CLASS_PERSON]
    people = sorted(people, key=lambda det: (det.box[0], det.box[1]))
    return [evaluate_person(person, detections, person_id=index) for index, person in enumerate(people, start=1)]


def draw_results(image, results: list[dict]):
    height, width = image.shape[:2]
    for result in results:
        person_box = clip_box(result["person_box"], width, height)
        head_box = clip_box(result["head_box"], width, height)
        torso_box = clip_box(result["torso_box"], width, height)

        helmet_ok = result["helmet"]
        vest_ok = result["vest"]
        person_color = (0, 200, 0) if helmet_ok and vest_ok else (0, 0, 255)
        head_color = (255, 200, 0) if helmet_ok else (0, 165, 255)
        torso_color = (255, 0, 0) if vest_ok else (0, 255, 255)

        cv2.rectangle(image, person_box[:2], person_box[2:], person_color, 2)
        cv2.rectangle(image, head_box[:2], head_box[2:], head_color, 1)
        cv2.rectangle(image, torso_box[:2], torso_box[2:], torso_color, 1)

        label = f"ID {result['person_id']} | helmet={helmet_ok} vest={vest_ok}"
        cv2.putText(
            image,
            label,
            (person_box[0], max(20, person_box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            person_color,
            2,
            cv2.LINE_AA,
        )

    return image
