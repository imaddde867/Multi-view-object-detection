from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, VideoInfo(path=path, width=width, height=height, fps=fps, frame_count=frame_count)


def iter_frames(
    cap: cv2.VideoCapture,
    *,
    stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, "cv2.typing.MatLike"]]:
    stride = max(1, int(stride))
    kept = 0
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        yield frame_idx, frame
        kept += 1
        if max_frames is not None and kept >= max_frames:
            break

