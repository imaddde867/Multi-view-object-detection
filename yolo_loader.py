"""Utility code for reading YOLO label files and exposing detections per camera."""
from __future__ import annotations

import ast
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2


@dataclass
class YoloDetection:
    """Single YOLO detection converted to pixel coordinates."""

    frame_id: int
    camera_idx: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    center: Tuple[float, float]

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]


def load_class_names(data_yaml_path: str) -> List[str]:
    """Parse the YOLO data.yaml file (only the 'names' field)."""

    default_names = ["person", "car", "bus"]

    if not os.path.exists(data_yaml_path):
        return default_names

    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("names:"):
                    _, value = stripped.split(":", 1)
                    value = value.strip()
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, Sequence):
                            return [str(v) for v in parsed]
                    except (ValueError, SyntaxError):
                        pass
    except OSError:
        return default_names

    return default_names


class YoloDetectionLoader:
    """Load YOLO detections for specific frames and cameras."""

    def __init__(
        self,
        labels_dir: str,
        image_root: str,
        class_names: Optional[Sequence[str]] = None,
        camera_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.labels_dir = labels_dir
        self.image_root = image_root
        self.class_names = list(class_names) if class_names else []
        self.camera_indices = list(camera_indices) if camera_indices else list(range(6))
        self.image_shapes = self._discover_image_shapes()

        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

    def available_frames(self) -> List[int]:
        pattern = os.path.join(self.labels_dir, "*.txt")
        frames = set()
        for label_path in glob.glob(pattern):
            name = os.path.basename(label_path)
            if "_c" not in name:
                continue
            frame_part = name.split("_c")[0]
            try:
                frames.add(int(frame_part))
            except ValueError:
                continue
        return sorted(frames)

    def load_frame(self, frame_id: int) -> Dict[int, List[YoloDetection]]:
        frame_str = f"{int(frame_id):08d}"
        detections: Dict[int, List[YoloDetection]] = {}

        for cam_idx in self.camera_indices:
            label_path = os.path.join(self.labels_dir, f"{frame_str}_c{cam_idx}.txt")
            if not os.path.exists(label_path):
                continue

            parsed = self._parse_label_file(label_path, frame_id, cam_idx)
            if parsed:
                detections[cam_idx] = parsed

        return detections

    # ------------------------------------------------------------------
    def _discover_image_shapes(self) -> Dict[int, Tuple[int, int]]:
        shapes: Dict[int, Tuple[int, int]] = {}

        for cam_idx in self.camera_indices:
            cam_dir = os.path.join(self.image_root, f"c{cam_idx}")
            if not os.path.isdir(cam_dir):
                continue

            sample_images = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
            if not sample_images:
                continue

            sample = cv2.imread(sample_images[0])
            if sample is None:
                continue

            shapes[cam_idx] = (sample.shape[1], sample.shape[0])  # width, height

        if not shapes:
            raise RuntimeError("Unable to determine image shapes for any camera.")

        return shapes

    def _parse_label_file(
        self, label_path: str, frame_id: int, cam_idx: int
    ) -> List[YoloDetection]:
        width, height = self._image_shape_for_camera(cam_idx)
        detections: List[YoloDetection] = []

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    detection = self._parse_detection_line(
                        line.strip(), frame_id, cam_idx, width, height
                    )
                    if detection:
                        detections.append(detection)
        except OSError:
            return []

        return detections

    def _parse_detection_line(
        self,
        line: str,
        frame_id: int,
        cam_idx: int,
        width: int,
        height: int,
    ) -> Optional[YoloDetection]:
        if not line:
            return None

        parts = line.split()
        if len(parts) < 5:
            return None

        try:
            class_id = int(float(parts[0]))
        except ValueError:
            return None

        conf = 1.0
        floats: List[float]

        try:
            if len(parts) == 5:
                floats = list(map(float, parts[1:5]))
            else:
                conf = float(parts[1])
                floats = list(map(float, parts[2:6]))
        except ValueError:
            return None

        if len(floats) != 4:
            return None

        x_center_norm, y_center_norm, w_norm, h_norm = floats

        x_center = x_center_norm * width
        y_center = y_center_norm * height
        box_width = w_norm * width
        box_height = h_norm * height

        x1 = max(0.0, x_center - box_width / 2.0)
        y1 = max(0.0, y_center - box_height / 2.0)
        x2 = min(float(width), x_center + box_width / 2.0)
        y2 = min(float(height), y_center + box_height / 2.0)

        class_name = (
            self.class_names[class_id]
            if 0 <= class_id < len(self.class_names)
            else str(class_id)
        )

        return YoloDetection(
            frame_id=frame_id,
            camera_idx=cam_idx,
            class_id=class_id,
            class_name=class_name,
            confidence=conf,
            bbox_xyxy=(x1, y1, x2, y2),
            center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        )

    def _image_shape_for_camera(self, cam_idx: int) -> Tuple[int, int]:
        if cam_idx in self.image_shapes:
            return self.image_shapes[cam_idx]
        if not self.image_shapes:
            raise RuntimeError("Image shapes not initialized")
        # Fallback: return first known shape
        return next(iter(self.image_shapes.values()))

