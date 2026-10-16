"""End-to-end tests for the dataset layout contract through verify_dataset."""

from __future__ import annotations

import json

import pytest

from multiview.dataset.layout import frame_name
from multiview.dataset.verify import verify_dataset


def _write_dataset(root, *, groups, frames, subset="train", with_labels=True):
    """frames: list of (group, cam, idx)."""
    images = root / subset / "images"
    labels = root / subset / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for group, cam, idx in frames:
        name = frame_name(group, cam, idx)
        (images / f"{name}.jpg").write_bytes(b"")
        if with_labels:
            (labels / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "meta.json").write_text(json.dumps({"groups": groups}))
    (root / "dataset.yaml").write_text(
        "names:\n  - person\n  - car\nnc: 2\n" f"{subset}: {subset}/images\n"
    )
    return root / "dataset.yaml"


class TestVerifyPairContract:
    def test_complete_group_passes(self, tmp_path) -> None:
        yaml_path = _write_dataset(
            tmp_path,
            groups={"g1": {"cam1": "a.mp4", "cam2": "b.mp4"}},
            frames=[("g1", "cam1", 0), ("g1", "cam2", 0), ("g1", "cam1", 5), ("g1", "cam2", 5)],
        )
        report = verify_dataset(yaml_path)
        assert report.errors == []
        assert report.images == 4
        assert report.boxes_total == 4

    def test_missing_camera_for_frame_errors(self, tmp_path) -> None:
        yaml_path = _write_dataset(
            tmp_path,
            groups={"g1": {"cam1": "a.mp4", "cam2": "b.mp4"}},
            frames=[("g1", "cam1", 0), ("g1", "cam2", 0), ("g1", "cam1", 5)],  # cam2 f5 missing
        )
        report = verify_dataset(yaml_path)
        assert any("Missing cams for g1 f000005" in e for e in report.errors)

    def test_same_frame_in_two_subsets_errors(self, tmp_path) -> None:
        yaml_path = _write_dataset(
            tmp_path,
            groups={"g1": {"cam1": "a.mp4", "cam2": "b.mp4"}},
            frames=[("g1", "cam1", 0), ("g1", "cam2", 0)],
        )
        # move cam1 f0 into val: the same (group, frame) now spans two subsets
        (tmp_path / "val" / "images").mkdir(parents=True)
        (tmp_path / "val" / "labels").mkdir(parents=True)
        (tmp_path / "val" / "images" / (frame_name("g1", "cam1", 0) + ".jpg")).write_bytes(b"")
        (tmp_path / "val" / "labels" / (frame_name("g1", "cam1", 0) + ".txt")).write_text(
            "0 0.5 0.5 0.2 0.2\n"
        )
        yaml_path.write_text("names:\n  - person\n  - car\nnc: 2\ntrain: train/images\nval: val/images\n")
        report = verify_dataset(yaml_path)
        assert any("appears in multiple subsets" in e for e in report.errors)

    def test_foreign_files_are_ignored(self, tmp_path) -> None:
        yaml_path = _write_dataset(
            tmp_path,
            groups={"g1": {"cam1": "a.mp4"}},
            frames=[("g1", "cam1", 3)],
        )
        # foreign files still need labels (verify's image/label pairing), but
        # they must not trip the multi-view pair check
        (tmp_path / "train" / "images" / "photo.jpg").write_bytes(b"")
        (tmp_path / "train" / "labels" / "photo.txt").write_text("")
        (tmp_path / "train" / "images" / "g1_cam1_fabc.jpg").write_bytes(b"")
        (tmp_path / "train" / "labels" / "g1_cam1_fabc.txt").write_text("")
        report = verify_dataset(yaml_path)
        assert report.errors == []

    def test_expected_names_mismatch_errors(self, tmp_path) -> None:
        yaml_path = _write_dataset(
            tmp_path,
            groups={"g1": {"cam1": "a.mp4"}},
            frames=[("g1", "cam1", 0)],
        )
        report = verify_dataset(yaml_path, expected_names=["person"])
        assert any("Class names mismatch" in e for e in report.errors)
