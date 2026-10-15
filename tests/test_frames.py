"""Tests for the shared lockstep frame iterator (C5).

Caps are duck-typed so no video files or cv2 handles are needed: each
fake script is a list of (frame_marker, timestamp_ms).
"""

from __future__ import annotations

import pytest

from multiview.frames import iter_group_frames


class FakeCap:
    def __init__(self, script: list[tuple[str, float]]):
        self.script = list(script)
        self.read_calls = 0
        self._last: tuple[bool, tuple[str, float]] = (False, ("", 0.0))

    def read(self) -> tuple[bool, object]:
        self.read_calls += 1
        if not self.script:
            return False, None
        frame = self.script.pop(0)
        self._last = (True, frame)
        return True, frame[0]

    def get(self, prop) -> float:
        # emulate cv2.CAP_PROP_POS_MSEC after a read: timestamp of last frame
        return self._last[1][1]


class TestNaiveAlign:
    def test_all_cams_every_raw_frame(self) -> None:
        caps = {
            "cam1": FakeCap([("a0", 0), ("a1", 10), ("a2", 20)]),
            "cam2": FakeCap([("b0", 5), ("b1", 15), ("b2", 25)]),
        }
        out = list(iter_group_frames(caps, align="naive"))
        assert [idx for idx, _ in out] == [0, 1, 2]
        assert out[1][1] == {"cam1": "a1", "cam2": "b1"}

    def test_stride_skips_raw_frames(self) -> None:
        caps = {"cam1": FakeCap([(f"a{i}", i * 10) for i in range(4)])}
        out = list(iter_group_frames(caps, align="naive", stride=2))
        assert [idx for idx, _ in out] == [0, 2]
        assert [f["cam1"] for _, f in out] == ["a0", "a2"]

    def test_max_frames_caps_kept_frames(self) -> None:
        caps = {"cam1": FakeCap([(f"a{i}", i * 10) for i in range(6)])}
        out = list(iter_group_frames(caps, align="naive", max_frames=2))
        assert len(out) == 2

    def test_stops_when_any_cam_ends(self) -> None:
        caps = {
            "cam1": FakeCap([(f"a{i}", i * 10) for i in range(4)]),
            "cam2": FakeCap([("b0", 5), ("b1", 15)]),
        }
        out = list(iter_group_frames(caps, align="naive"))
        assert len(out) == 2


class TestTimestampAlign:
    def test_other_cams_advance_to_sync_timestamp(self) -> None:
        sync = FakeCap([("s0", 100.0), ("s1", 300.0)])
        other = FakeCap([("o0", 100.0), ("o1", 200.0), ("o2", 400.0)])
        caps = {"sync": sync, "other": other}
        out = list(iter_group_frames(caps, align="timestamp", sync_cam="sync"))
        assert [idx for idx, _ in out] == [0, 1]
        assert out[0][1] == {"sync": "s0", "other": "o0"}
        # sync ts=300 skips other's ts=200 and takes ts=400
        assert out[1][1] == {"sync": "s1", "other": "o2"}

    def test_stops_when_other_cam_runs_out(self) -> None:
        sync = FakeCap([(f"s{i}", (i + 1) * 100.0) for i in range(3)])
        other = FakeCap([("o0", 100.0)])
        caps = {"sync": sync, "other": other}
        out = list(iter_group_frames(caps, align="timestamp", sync_cam="sync"))
        assert len(out) == 1

    def test_stride_applies_to_sync_frames(self) -> None:
        sync = FakeCap([(f"s{i}", (i + 1) * 100.0) for i in range(4)])
        other = FakeCap([(f"o{i}", (i + 1) * 100.0) for i in range(4)])
        caps = {"sync": sync, "other": other}
        out = list(iter_group_frames(caps, align="timestamp", sync_cam="sync", stride=2))
        assert [idx for idx, _ in out] == [0, 2]


def test_no_caps_yields_nothing() -> None:
    assert list(iter_group_frames({}, align="naive")) == []
