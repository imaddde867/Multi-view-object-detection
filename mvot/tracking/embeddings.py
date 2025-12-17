from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


class Embedder(Protocol):
    def embed(self, frame_bgr: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
        ...


def _crop(frame_bgr: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1]


@dataclass(frozen=True)
class ColorHistEmbedder:
    bins: int = 16

    def embed(self, frame_bgr: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
        crop = _crop(frame_bgr, xyxy)
        if crop is None:
            return np.zeros((self.bins * 3,), dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180]).reshape(-1)
        s_hist = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256]).reshape(-1)
        v_hist = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256]).reshape(-1)
        feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = float(np.linalg.norm(feat) + 1e-9)
        return feat / norm


class TorchResNet18Embedder:
    def __init__(self, *, device: str = "cpu"):
        try:
            import torch
            import torchvision.transforms as T
            from torchvision.models import resnet18
        except Exception as e:  # pragma: no cover
            raise RuntimeError("TorchResNet18Embedder requires torch+torchvision.") from e

        self.torch = torch
        self.device = device
        model = resnet18(weights="DEFAULT")
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(device)
        self.model = model
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def embed(self, frame_bgr: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
        crop = _crop(frame_bgr, xyxy)
        if crop is None:
            return np.zeros((512,), dtype=np.float32)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model(x).squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(feat) + 1e-9)
        return feat / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False).reshape(-1)
    b = b.astype(np.float32, copy=False).reshape(-1)
    na = float(np.linalg.norm(a) + 1e-9)
    nb = float(np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / (na * nb))

