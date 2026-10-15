"""Dataset naming and layout contract.

The labeling pipeline writes frames out under a name and directory layout
that the dataset verifier reads back. Both directions used to hard-code
that format in two separate places; this module is the single definition.
"""

from __future__ import annotations

from pathlib import Path


def frame_name(group: str, cam: str, frame_idx: int) -> str:
    """Canonical name of one camera frame inside a group dataset."""
    return f"{group}_{cam}_f{frame_idx:06d}"


def frame_prefix(group: str, cam: str) -> str:
    """The stem prefix all of a camera's frame names start with."""
    return f"{group}_{cam}_f"


def split_frame_name(stem: str) -> tuple[str, str]:
    """Split a frame name into (prefix, index tail).

    'g1_cam1_f000123' -> ('g1_cam1_f', '000123'). The prefix keeps the '_f'
    marker, so it matches frame_prefix(group, cam) exactly and
    prefix + tail reconstructs the name. Stems without a '_f' split return
    (stem, ''); non-digit tails come back verbatim so the caller can apply
    its own reporting policy. The split is on the LAST '_f'.
    """
    if "_f" not in stem:
        return stem, ""
    prefix, tail = stem.rsplit("_f", 1)
    return f"{prefix}_f", tail


def subset_images(root: Path, subset: str) -> Path:
    return root / subset / "images"


def subset_labels(root: Path, subset: str) -> Path:
    return root / subset / "labels"


def sibling_labels(images_dir: Path) -> Path:
    """The labels dir implied by an images dir (the YAML layout convention)."""
    return images_dir.parent / "labels"
