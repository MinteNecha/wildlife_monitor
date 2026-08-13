"""
MegaDetector multi-instance animal detector (Package P2).

MegaDetector (Microsoft AI for Earth) is a YOLOv5-based detector trained
specifically on camera trap images to find every animal in a frame,
regardless of species. It returns one bounding box per detected individual
with a confidence score.

In Pipeline 1c (BioCLIP + MegaDetector) MegaDetector is used AFTER BioCLIP
has already identified and ranked the images. Its job is localisation and
counting — finding how many individual animals are present in each selected
image. This instance count is the primary input to Pipeline 2's social
structure classification.

MegaDetector outputs three class labels: animal (1), vehicle (2), person (3).
Only class 1 (animal) detections are returned — vehicles and people are
filtered out because they are noise in a species-monitoring context.

Installation
------------
pip install megadetector  # Microsoft AI for Earth package
# OR use the Hugging Face version:
pip install huggingface_hub
# Then the model downloads automatically on first use.

The checkpoint (~700 MB) downloads automatically to the HuggingFace cache
on first run. No manual download required.
"""

from __future__ import annotations

import numpy as np

from wildlife_monitor.config import MEGADETECTOR_CONF, SystemConfig

# MegaDetector class ID for "animal"
_ANIMAL_CLASS = 1


class MegaDetector:
    """
    Finds ALL animals in a camera trap image and returns their boxes.

    Unlike YOLO (which returns one best box) MegaDetector returns every
    detected individual above the confidence threshold. The count of returned
    boxes is the instance_count stored in DetectionRecord.

    Returns
    -------
    boxes  : list of (x1, y1, x2, y2, confidence) tuples — one per individual
    count  : number of animals detected
    """

    def __init__(self) -> None:
        print("[INFO] Loading MegaDetector ...")
        try:
            from megadetector.detection.run_detector import load_detector
            self._model = load_detector("MDV5A")
            self.backend = "megadetector"
            print("[INFO] MegaDetector ready.")
        except ImportError:
            print("[WARN] megadetector package not installed. "
                  "Falling back to YOLO multi-detection mode.")
            self._model = None
            self.backend = "yolo_fallback"
            self._load_yolo_fallback()

    def _load_yolo_fallback(self) -> None:
        """
        Fallback when megadetector is not installed.
        Uses YOLOv11 with NMS disabled to return all detections
        rather than just the single best one.
        """
        from ultralytics import YOLO
        from wildlife_monitor.config import YOLO_MODEL
        self._yolo = YOLO(YOLO_MODEL)
        print("[INFO] YOLO fallback loaded for multi-detection.")

    def detect_all(
        self, image_rgb: np.ndarray
    ) -> tuple[list[tuple[int, int, int, int, float]], int]:
        """
        Return all animal detections in the image.

        Returns
        -------
        boxes : list of (x1, y1, x2, y2, confidence) — one per animal
        count : number of animals detected
        """
        if self.backend == "megadetector":
            return self._detect_megadetector(image_rgb)
        return self._detect_yolo_fallback(image_rgb)

    def _detect_megadetector(
        self, image_rgb: np.ndarray
    ) -> tuple[list[tuple[int, int, int, int, float]], int]:
        """Run MegaDetector and return all animal boxes."""
        import cv2
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result = self._model.generate_detections_one_image(
            image_bgr, image_id="frame", detection_threshold=MEGADETECTOR_CONF
        )
        height, width = image_rgb.shape[:2]
        boxes = []
        for det in result.get("detections", []):
            if det.get("category") != str(_ANIMAL_CLASS):
                continue
            conf = float(det["conf"])
            # MegaDetector returns normalised [x, y, w, h] (0-1 fractions)
            x, y, w, h = det["bbox"]
            x1 = int(x * width)
            y1 = int(y * height)
            x2 = int((x + w) * width)
            y2 = int((y + h) * height)
            boxes.append((x1, y1, x2, y2, conf))

        return boxes, len(boxes)

    def _detect_yolo_fallback(
        self, image_rgb: np.ndarray
    ) -> tuple[list[tuple[int, int, int, int, float]], int]:
        """YOLO fallback — returns all animal-class detections."""
        import cv2
        from wildlife_monitor.config import ANIMAL_COCO_IDS
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self._yolo(image_bgr, verbose=False, conf=MEGADETECTOR_CONF)[0]
        bx = results.boxes
        if bx is None or len(bx) == 0:
            return [], 0

        confidences = bx.conf.cpu().numpy()
        class_ids   = bx.cls.cpu().numpy().astype(int)
        coordinates = bx.xyxy.cpu().numpy().astype(int)

        boxes = []
        for i in range(len(bx)):
            if class_ids[i] not in ANIMAL_COCO_IDS:
                continue
            x1, y1, x2, y2 = coordinates[i]
            boxes.append((int(x1), int(y1), int(x2), int(y2),
                          float(confidences[i])))

        return boxes, len(boxes)
