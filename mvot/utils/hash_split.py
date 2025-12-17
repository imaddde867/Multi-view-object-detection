from __future__ import annotations

import hashlib


def stable_split(name: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0 + 1e-9:
        raise ValueError("Expected train_ratio>=0, val_ratio>=0 and train_ratio+val_ratio<=1")
    key = f"{seed}:{name}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()
    r = int(h[:8], 16) / 0xFFFFFFFF
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"

