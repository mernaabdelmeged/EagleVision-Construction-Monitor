"""
detector.py — YOLOv8-based construction equipment detector
Detects: excavators, trucks, bulldozers etc.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs relevant to construction equipment
CONSTRUCTION_CLASSES = {
    7: "truck",
    8: "bus",         # sometimes misdetected heavy equipment
    16: "horse",      # fallback (not used)
}

# We'll use a broader approach - detect all vehicles and filter
VEHICLE_CLASS_IDS = {7, 8}  # truck classes in COCO

# Labels we'll use for display
EQUIPMENT_LABEL_MAP = {
    "truck": "truck",
    "excavator": "excavator",
    "bulldozer": "bulldozer",
}


class EquipmentDetector:
    """
    Wraps YOLOv8 for construction equipment detection.
    Uses YOLOv8n (nano) by default for speed, can switch to YOLOv8x for accuracy.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.3):
        """
        Args:
            model_path: Path to YOLOv8 weights. Downloads automatically if not present.
            conf_threshold: Minimum confidence to keep a detection.
        """
        print(f"[Detector] Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # {0: 'person', 7: 'truck', ...}
        print(f"[Detector] Model loaded. Classes: {len(self.class_names)}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single frame.

        Returns:
            List of dicts:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'class_name': str,
            }
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Filter to construction-relevant classes only
                # Accept trucks, buses (heavy machinery misclassified), and person
                # In practice, use a fine-tuned model for excavators
                class_name = self.class_names.get(class_id, "unknown")

                # Keep all vehicle-like detections for construction sites
                # Skip persons, animals, furniture etc.
                skip_classes = {
                    "person", "bicycle", "motorcycle", "airplane", "bird",
                    "cat", "dog", "horse", "sheep", "cow", "elephant",
                    "bear", "zebra", "giraffe", "chair", "couch", "bed",
                }
                if class_name in skip_classes:
                    continue

                # Minimum bounding box size filter (ignore tiny detections)
                w = x2 - x1
                h = y2 - y1
                if w < 40 or h < 40:
                    continue

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name,
                })

        return detections

    def draw_raw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw raw detections (before tracking) on frame for debugging."""
        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return vis
