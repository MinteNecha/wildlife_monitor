"""
YOLOv11 detector (Package P2).

Wraps the Ultralytics YOLOv11 nano model. Provides two detection modes:

    detect()     — returns the single highest-confidence animal box.
                   Used by Pipeline 1b for the speed-oriented single-box result.

    detect_all() — returns ALL animal boxes above the confidence threshold.
                   Used by Pipeline 1b's multi-instance mode to provide
                   instance_count comparable to MegaDetector (Pipeline 1c).

The two-tier animal filter prefers COCO animal classes (IDs 16-25) over
other classes. For Serengeti species not in COCO (buffalo, wildebeest, lion,
cheetah, hyena) the fallback uses the highest-confidence detection of any
class — the box location is still correct even when the class label is wrong.
"""

from __future__ import annotations

import numpy as np

from wildlife_monitor.config import YOLO_MODEL, ANIMAL_COCO_IDS


class YOLODetector:
    """Detects animal bounding boxes in an image."""

    def __init__(self) -> None:
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLOv11 ({YOLO_MODEL}) ...")
        self.model = YOLO(YOLO_MODEL)
        print("[INFO] YOLOv11 ready.")

    def detect(
        self, image_bgr: np.ndarray
    ) -> tuple[int, int, int, int, float] | None:
        """
        Return (x1, y1, x2, y2, confidence) for the single best animal,
        or None if nothing detected.

        Animal-class detections are preferred; if none present the
        highest-confidence detection of any class is returned.
        """
        results = self.model(image_bgr, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        confidences  = boxes.conf.cpu().numpy()
        class_ids    = boxes.cls.cpu().numpy().astype(int)
        coordinates  = boxes.xyxy.cpu().numpy().astype(int)

        animal_mask = np.isin(class_ids, list(ANIMAL_COCO_IDS))
        if animal_mask.any():
            candidates = np.where(animal_mask)[0]
            best = candidates[np.argmax(confidences[candidates])]
        else:
            best = int(np.argmax(confidences))

        x1, y1, x2, y2 = coordinates[best]
        return int(x1), int(y1), int(x2), int(y2), float(confidences[best])

    def detect_all(
        self, image_bgr: np.ndarray, conf_threshold: float = 0.20
    ) -> tuple[list[tuple[int, int, int, int, float]], int]:
        """
        Return ALL animal detections above conf_threshold.

        Returns
        -------
        boxes : list of (x1, y1, x2, y2, confidence) — one per animal
        count : number of animals detected

        All COCO animal-class detections are included. When no animal class
        is found, falls back to the single best detection of any class
        (count=1) so the pipeline always returns something.
        """
        results = self.model(image_bgr, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return [], 0

        confidences = boxes.conf.cpu().numpy()
        class_ids   = boxes.cls.cpu().numpy().astype(int)
        coordinates = boxes.xyxy.cpu().numpy().astype(int)

        # Collect all animal-class detections above threshold
        output = []
        for i in range(len(boxes)):
            if class_ids[i] in ANIMAL_COCO_IDS and confidences[i] >= conf_threshold:
                x1, y1, x2, y2 = coordinates[i]
                output.append((int(x1), int(y1), int(x2), int(y2),
                               float(confidences[i])))

        # Fallback: no animal class found — return single best detection
        if not output:
            best = int(np.argmax(confidences))
            x1, y1, x2, y2 = coordinates[best]
            output.append((int(x1), int(y1), int(x2), int(y2),
                          float(confidences[best])))

        return output, len(output)
