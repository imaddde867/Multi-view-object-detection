"""Tests for the dataset naming + layout contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from multiview.dataset.layout import (
    frame_name,
    frame_prefix,
    sibling_labels,
    split_frame_name,
    subset_images,
    subset_labels,
)


class TestFrameNames:
    def test_round_trip(self) -> None:
        name = frame_name("g34", "cam1", 123)
        assert name == "g34_cam1_f000123"
        prefix, idx = split_frame_name(name)
        assert prefix == frame_prefix("g34", "cam1")
        assert prefix + idx == name

    def test_prefix_matches_name_head(self) -> None:
        assert frame_prefix("g34", "cam1") == "g34_cam1_f"
        assert frame_name("g34", "cam1", 7).startswith(frame_prefix("g34", "cam1"))

    def test_non_frame_stem(self) -> None:
        assert split_frame_name("random_image") == ("random_image", "")

    def test_bad_suffix_tail_comes_back_verbatim(self) -> None:
        # The caller decides what to do with a non-digit tail (verify warns)
        assert split_frame_name("g34_cam1_fabc") == ("g34_cam1_f", "abc")

    def test_empty_index_suffix(self) -> None:
        assert split_frame_name("g34_cam1_f") == ("g34_cam1_f", "")

    def test_zero_padded_index_survives(self) -> None:
        prefix, idx = split_frame_name(frame_name("g", "c", 5))
        assert idx == "000005"


class TestLayoutDirs:
    def test_subset_dirs(self) -> None:
        root = Path("data/processed/ds")
        assert subset_images(root, "train") == root / "train" / "images"
        assert subset_labels(root, "train") == root / "train" / "labels"

    def test_sibling_labels_contract(self) -> None:
        # verify derives the labels dir from wherever the yaml points images
        images_dir = Path("ds/train/images")
        assert sibling_labels(images_dir) == Path("ds/train/labels")
