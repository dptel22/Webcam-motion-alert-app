"""Lightweight CPU-friendly Deep Learning object detector using YOLOv8."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """Single object detection produced by a DL detector or CVIP contour extractor."""

    bbox: np.ndarray  # [x1, y1, x2, y2] float32
    confidence: float
    class_id: int
    class_name: str
    embedding: np.ndarray | None = None


class YOLODetector:
    """Lazy-loading YOLOv8 detector wrapper tuned for real-time CPU inference."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a BGR frame and return Detection objects."""
        model = self._ensure_model()
        results = model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names or {}
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = boxes.conf.cpu().numpy().astype(float)
        clss = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for box, conf, cls_id in zip(xyxy, confs, clss):
            detections.append(
                Detection(
                    bbox=box,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=str(names.get(int(cls_id), f"class_{cls_id}")),
                )
            )
        return detections
