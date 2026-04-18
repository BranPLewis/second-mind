import os
import hashlib
import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "/model/weights/best.pt"
DEFAULT_CONFIDENCE = 0.01


@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]


class ObjectDetector:
    def __init__(
        self, model_path: str | None = None, confidence: float = DEFAULT_CONFIDENCE
    ):
        self.model_path = model_path or os.getenv("YOLO_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.confidence = confidence
        self.model = self._load_model(self.model_path)

    @staticmethod
    def _load_model(model_path: str):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"YOLO model not found at '{model_file}'. Set YOLO_MODEL_PATH to a valid weights file."
            )
        return YOLO(str(model_file))

    @staticmethod
    def _label_color_bgr(label: str) -> tuple[int, int, int]:
        digest = hashlib.md5(label.encode("utf-8")).hexdigest()
        hue_seed = int(digest[:6], 16) / float(0xFFFFFF)
        r, g, b = colorsys.hsv_to_rgb(hue_seed, 0.85, 1.0)
        return int(b * 255), int(g * 255), int(r * 255)

    @classmethod
    def label_color_hex(cls, label: str) -> str:
        b, g, r = cls._label_color_bgr(label)
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _distinct_palette_bgr() -> list[tuple[int, int, int]]:
        rgb_palette = [
            (255, 99, 71),
            (65, 182, 230),
            (255, 193, 7),
            (123, 97, 255),
            (46, 204, 113),
            (255, 127, 80),
            (233, 30, 99),
            (0, 188, 212),
            (156, 204, 101),
            (255, 152, 0),
            (121, 85, 72),
            (0, 150, 136),
            (63, 81, 181),
            (244, 67, 54),
            (139, 195, 74),
            (33, 150, 243),
            (255, 87, 34),
            (103, 58, 183),
            (205, 220, 57),
            (0, 121, 107),
        ]
        return [(b, g, r) for r, g, b in rgb_palette]

    @classmethod
    def build_label_color_map(
        cls, labels: list[str]
    ) -> dict[str, tuple[int, int, int]]:
        unique_labels = sorted(set(labels))
        if not unique_labels:
            return {}

        palette = cls._distinct_palette_bgr()
        color_map: dict[str, tuple[int, int, int]] = {}

        for idx, label in enumerate(unique_labels):
            if idx < len(palette):
                color_map[label] = palette[idx]
                continue

            hue = (idx * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.98)
            color_map[label] = (int(b * 255), int(g * 255), int(r * 255))

        return color_map

    @staticmethod
    def bgr_to_hex(color_bgr: tuple[int, int, int]) -> str:
        b, g, r = color_bgr
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _all_detections(result: Any) -> list[DetectionResult]:
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[DetectionResult] = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_center = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names.get(cls_id, str(cls_id))
            detections.append(
                DetectionResult(
                    label=label,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=box_center,
                )
            )

        return detections

    @staticmethod
    def highest_confidence_per_label(
        detections: list[DetectionResult],
    ) -> dict[str, DetectionResult]:
        best_by_label: dict[str, DetectionResult] = {}
        for detection in detections:
            current_best = best_by_label.get(detection.label)
            if current_best is None or detection.confidence > current_best.confidence:
                best_by_label[detection.label] = detection
        return best_by_label

    def detect_components(self, frame_bgr):
        results = self.model(frame_bgr, conf=self.confidence, verbose=False)
        result = results[0]
        detections = self._all_detections(result)
        return detections, result

    def detect_best_component_by_label(self, frame_bgr, label: str):
        detections, result = self.detect_components(frame_bgr)
        best_for_label = self.highest_confidence_per_label(detections).get(label)
        return best_for_label, detections, result

    def detect_center_object(self, frame_bgr):
        detections, result = self.detect_components(frame_bgr)
        if not detections:
            return None, result
        frame_center = (frame_bgr.shape[1] // 2, frame_bgr.shape[0] // 2)
        nearest = min(
            detections,
            key=lambda d: (
                (d.center[0] - frame_center[0]) ** 2
                + (d.center[1] - frame_center[1]) ** 2
            ),
        )
        return nearest, result

    @staticmethod
    def draw_detection(
        frame_bgr,
        detection: DetectionResult | None,
        reference_point: tuple[int, int] | None = None,
    ):
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        if reference_point is None:
            reference_point = (w // 2, h // 2)

        cv2.drawMarker(
            annotated,
            reference_point,
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )

        if detection is None:
            cv2.putText(
                annotated,
                "No object detected near selected point",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return annotated

        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.circle(annotated, detection.center, 5, (60, 220, 60), -1)

        label = f"{detection.label} ({detection.confidence:.3f})"
        cv2.putText(
            annotated,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (60, 220, 60),
            2,
            cv2.LINE_AA,
        )
        return annotated

    @staticmethod
    def draw_all_detections(
        frame_bgr,
        detections: list[DetectionResult],
        color_map: dict[str, tuple[int, int, int]] | None = None,
    ):
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        line_thickness = max(3, int(min(h, w) / 220))

        if not detections:
            return annotated

        colors = color_map or ObjectDetector.build_label_color_map(
            [d.label for d in detections]
        )

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors.get(
                detection.label, ObjectDetector._label_color_bgr(detection.label)
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        return annotated
