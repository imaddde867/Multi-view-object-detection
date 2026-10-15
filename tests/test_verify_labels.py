"""Label-syntax validation coverage for verify_dataset (audit gap #1)."""

from __future__ import annotations

import json

from multiview.dataset.layout import frame_name
from multiview.dataset.verify import verify_dataset


def _dataset_with_label(tmp_path, label_text: str, stem: str | None = None) -> str:
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    name = stem or frame_name("g1", "cam1", 0)
    (images / f"{name}.jpg").write_bytes(b"")
    (labels / f"{name}.txt").write_text(label_text)
    (tmp_path / "dataset.yaml").write_text("names:\n  - person\n  - car\nnc: 2\ntrain: train/images\n")
    return str(tmp_path / "dataset.yaml")


def _report(yaml_path: str):
    return verify_dataset(yaml_path, check_pairs=False)


def test_four_field_line_errors(tmp_path) -> None:
    yaml_path = _dataset_with_label(tmp_path, "0 0.5 0.5 0.2\n")
    assert any("expected 5 fields" in e for e in _report(yaml_path).errors)


def test_six_field_line_errors(tmp_path) -> None:
    yaml_path = _dataset_with_label(tmp_path, "0 0.5 0.5 0.2 0.2 0.1\n")
    assert any("expected 5 fields" in e for e in _report(yaml_path).errors)


def test_non_numeric_line_errors(tmp_path) -> None:
    yaml_path = _dataset_with_label(tmp_path, "0 x 0.5 0.2 0.2\n")
    assert any("invalid YOLO line" in e for e in _report(yaml_path).errors)


def test_class_id_out_of_range_errors(tmp_path) -> None:
    yaml_path = _dataset_with_label(tmp_path, "5 0.5 0.5 0.2 0.2\n")
    assert any("class id 5 out of range" in e for e in _report(yaml_path).errors)


def test_box_out_of_unit_range_errors(tmp_path) -> None:
    yaml_path = _dataset_with_label(tmp_path, "0 1.5 0.5 0.2 0.2\n")
    assert any("xc=1.5 out of [0,1]" in e for e in _report(yaml_path).errors)


def test_zero_size_box_errors(tmp_path) -> None:
    yaml_path = _dataset_with_label(tmp_path, "0 0.5 0.5 0.0 0.2\n")
    assert any("invalid box size" in e for e in _report(yaml_path).errors)


def test_image_without_label_errors(tmp_path) -> None:
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "solo.jpg").write_bytes(b"")
    (tmp_path / "dataset.yaml").write_text("names:\n  - person\nnc: 1\ntrain: train/images\n")
    assert any("missing label for image: solo.jpg" in e for e in _report(str(tmp_path / "dataset.yaml")).errors)


def test_label_without_image_warns(tmp_path) -> None:
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (labels / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (images / "solo.jpg").write_bytes(b"")
    (labels / "solo.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "dataset.yaml").write_text("names:\n  - person\nnc: 1\ntrain: train/images\n")
    report = _report(str(tmp_path / "dataset.yaml"))
    assert any("orphan" in w for w in report.warnings)
    assert report.errors == []


def test_missing_yaml_raises(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        verify_dataset(tmp_path / "nope.yaml")


def test_names_as_dict_is_accepted(tmp_path) -> None:
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    name = frame_name("g1", "cam1", 0)
    (images / f"{name}.jpg").write_bytes(b"")
    (labels / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "dataset.yaml").write_text(
        "names: {0: person, 1: car}\ntrain: train/images\n"
    )
    (tmp_path / "meta.json").write_text(json.dumps({}))
    report = _report(str(tmp_path / "dataset.yaml"))
    assert report.names == ["person", "car"]
