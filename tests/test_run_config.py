"""Tests for the typed config at the run pipeline seam."""

from __future__ import annotations

import numpy as np
import pytest

from multiview.system.run import RunConfig


def _minimal_cfg() -> dict:
    return {
        "cameras": {
            "cam1": {"source": "a.mp4"},
            "cam2": {"source": "b.mp4"},
        },
        "groups": {"g1": ["cam1", "cam2"]},
        "detector": {"model": "weights.pt"},
    }


class TestRunConfigDefaults:
    def test_defaults_fill_when_cfg_is_minimal(self) -> None:
        c = RunConfig.from_config(_minimal_cfg())
        assert c.targets == ["person", "car"]
        assert c.conf == pytest.approx(0.35)
        assert c.iou == pytest.approx(0.5)
        assert c.imgsz == 960
        assert c.max_det == 0
        assert c.embedder_type == "colorhist"
        assert c.embedder_bins == 16
        assert c.max_age == 30
        assert c.global_match_threshold == pytest.approx(0.75)
        assert c.frame_stride == 1
        assert c.max_frames is None
        assert str(c.out_dir) == "results/system"
        assert c.write_video is False
        assert c.debug_assoc is False

    def test_missing_model_raises(self) -> None:
        cfg = _minimal_cfg()
        del cfg["detector"]["model"]
        with pytest.raises(ValueError, match="detector.model"):
            RunConfig.from_config(cfg)

    def test_missing_cameras_or_groups_raises(self) -> None:
        cfg = _minimal_cfg()
        del cfg["groups"]
        with pytest.raises(ValueError, match="groups"):
            RunConfig.from_config(cfg)


class TestRunConfigCoercions:
    def test_targets_from_list_or_string(self) -> None:
        cfg = _minimal_cfg()
        cfg["detector"]["targets"] = "person, car ,bus"
        assert RunConfig.from_config(cfg).targets == ["person", "car", "bus"]
        cfg["detector"]["targets"] = ["person", "bus"]
        assert RunConfig.from_config(cfg).targets == ["person", "bus"]

    def test_source_map_string_parsed(self) -> None:
        cfg = _minimal_cfg()
        cfg["detector"]["source_map"] = "truck=car, motorcycle=car"
        assert RunConfig.from_config(cfg).source_map == {"truck": "car", "motorcycle": "car"}

    def test_max_frames_zero_means_none(self) -> None:
        cfg = _minimal_cfg()
        cfg.setdefault("run", {})["max_frames"] = 0
        c = RunConfig.from_config(cfg)
        assert c.max_frames is None
        cfg["run"]["max_frames"] = 12
        assert RunConfig.from_config(cfg).max_frames == 12

    def test_bool_coercion_unlike_builtin_bool(self) -> None:
        # bool("false") is True in Python; YAML strings must not hit that trap
        cfg = _minimal_cfg()
        cfg.setdefault("output", {})["write_video"] = "false"
        assert RunConfig.from_config(cfg).write_video is False
        cfg["output"]["write_video"] = "true"
        assert RunConfig.from_config(cfg).write_video is True
        cfg["output"]["write_video"] = True
        assert RunConfig.from_config(cfg).write_video is True

    def test_embedder_string_shorthand_and_digit_device(self) -> None:
        cfg = _minimal_cfg()
        cfg.setdefault("tracker", {})["embedder"] = "torch_resnet18"
        c = RunConfig.from_config(cfg)
        assert c.embedder_type == "torch_resnet18"
        cfg["tracker"]["embedder"] = {"type": "torch_resnet18", "device": "0"}
        c = RunConfig.from_config(cfg)
        assert c.embedder_device == "cuda:0"
        cfg.setdefault("runtime", {})["device"] = "cpu"
        cfg["tracker"]["embedder"] = {"type": "torch_resnet18"}
        c = RunConfig.from_config(cfg)
        assert c.embedder_device == "cpu"


class TestRunConfigHomographies:
    def _two_cam(self) -> dict:
        cfg = _minimal_cfg()
        cfg["cameras"]["cam2"] = {
            "source": "b.mp4",
            "image_points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "world_points": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        }
        return cfg

    def test_homography_computed_from_points(self) -> None:
        c = RunConfig.from_config(self._two_cam())
        h = c.cam_homographies["cam2"]
        assert h.shape == (3, 3)
        assert h.dtype == np.float32

    def test_homography_matrix_passthrough(self) -> None:
        cfg = _minimal_cfg()
        cfg["cameras"]["cam1"] = {"source": "a.mp4", "homography": np.eye(3).tolist()}
        c = RunConfig.from_config(cfg)
        assert np.allclose(c.cam_homographies["cam1"], np.eye(3), atol=1e-6)

    def test_wrong_matrix_shape_raises(self) -> None:
        cfg = _minimal_cfg()
        cfg["cameras"]["cam1"] = {"source": "a.mp4", "homography": [[1.0, 0.0], [0.0, 1.0]]}
        with pytest.raises(ValueError, match="3x3"):
            RunConfig.from_config(cfg)

    def test_too_few_points_raises(self) -> None:
        cfg = _minimal_cfg()
        cfg["cameras"]["cam2"] = {
            "source": "b.mp4",
            "image_points": [[0.0, 0.0], [1.0, 1.0]],
            "world_points": [[0.0, 0.0], [1.0, 1.0]],
        }
        with pytest.raises(ValueError, match="N>=4"):
            RunConfig.from_config(cfg)

    def test_mismatched_point_shapes_raise(self) -> None:
        cfg = _minimal_cfg()
        cfg["cameras"]["cam2"] = {
            "source": "b.mp4",
            "image_points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "world_points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        }
        with pytest.raises(ValueError, match="matching shapes"):
            RunConfig.from_config(cfg)
