"""
YOLOv11 detector (Package P2).

Wraps the Ultralytics YOLOv11 nano model. It returns the bounding box of
the highest-confidence animal in an image, falling back to the highest-
confidence detection of any class when no COCO animal class matches — the
box is still useful even when the class label is wrong for a species YOLO
was never trained on.
"""

from __future__ import annotations

import numpy as np

from wildlife_monitor.config import YOLO_MODEL, ANIMAL_COCO_IDS


class YOLODetector:
    """Detects the primary animal bounding box in an image."""

    def __init__(self) -> None:
        from ultralytics import YOLO

        print(f"[INFO] Loading YOLOv11 ({YOLO_MODEL}) ...")
        self.model = YOLO(YOLO_MODEL)
        print("[INFO] YOLOv11 ready.")

    def detect(
        self, image_bgr: np.ndarray
    ) -> tuple[int, int, int, int, float] | None:
        """Return ``(x1, y1, x2, y2, confidence)`` or ``None`` if empty.

        Animal-class detections are preferred; if none are present the
        highest-confidence detection of any class is returned so that the
        localisation is still usable.
        """
        results = self.model(image_bgr, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        coordinates = boxes.xyxy.cpu().numpy().astype(int)

        animal_mask = np.isin(class_ids, list(ANIMAL_COCO_IDS))
        if animal_mask.any():
            candidates = np.where(animal_mask)[0]
            best = candidates[np.argmax(confidences[candidates])]
        else:
            best = int(np.argmax(confidences))

        x1, y1, x2, y2 = coordinates[best]
        return int(x1), int(y1), int(x2), int(y2), float(confidences[best])
