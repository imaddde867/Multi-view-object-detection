"""Tests for the deepened global ID assigner: one entry point, one call.

The run path used to replay a 4-step fallback (get → views_by_key →
ensure_view → ensure) to learn a global id, and the last fallback minted
ids with no track state, so the next frame re-minted them — an id that
could silently flip. assign_frame now returns the full mapping for every
view it was given, and debug output goes through a recorder seam instead
of a debug flag plus caller-side payload assembly.
"""

from __future__ import annotations

import numpy as np
import pytest

from multiview.tracking.global_association import GlobalIDAssigner, GlobalTrackView


def view(cam: str, local: int, emb=(1.0, 0.0), cls: int = 0, xyxy=(0.0, 0.0, 10.0, 10.0), world_xy=None) -> GlobalTrackView:
    return GlobalTrackView(
        cam=cam, local_id=local, cls_id=cls, xyxy=xyxy,
        embedding=np.asarray(emb, dtype=np.float32), world_xy=world_xy,
    )


class FakeRecorder:
    def __init__(self):
        self.calls = []

    def record(self, frame_idx, views_by_cam, payload):
        self.calls.append((frame_idx, views_by_cam, payload))


def _assigner(**kw) -> GlobalIDAssigner:
    params = {"match_threshold": 0.75, "max_age": 30}
    params.update(kw)
    return GlobalIDAssigner(**params)


class TestSingleEntryPoint:
    def test_mapping_returned_for_every_view(self) -> None:
        a = _assigner()
        views = {"cam1": [view("cam1", 1), view("cam1", 2, emb=(0.0, 1.0))], "cam2": [view("cam2", 9)]}
        mapping = a.assign_frame(0, views)
        assert set(mapping) == {("cam1", 1), ("cam1", 2), ("cam2", 9)}
        assert mapping[("cam1", 1)] == mapping[("cam2", 9)]  # same embedding+cls merges

    def test_ids_are_stable_across_frames(self) -> None:
        # The regression: a gid minted without track state was invisible to
        # association and got re-minted (flipped) on the next frame.
        a = _assigner()
        v = view("cam1", 1)
        first = a.assign_frame(0, {"cam1": [v]})
        for f in (1, 2, 5):
            again = a.assign_frame(f, {"cam1": [v]})
            assert again[("cam1", 1)] == first[("cam1", 1)], f"gid flipped at frame {f}"

    def test_reappearing_view_keeps_its_id(self) -> None:
        a = _assigner()
        v = view("cam1", 1)
        first = a.assign_frame(0, {"cam1": [v]})
        a.assign_frame(1, {"cam1": []})  # track absent
        back = a.assign_frame(2, {"cam1": [v]})
        assert back[("cam1", 1)] == first[("cam1", 1)]

    def test_appearance_change_breaks_association(self) -> None:
        a = _assigner()
        a.assign_frame(0, {"cam1": [view("cam1", 1)]})
        changed = a.assign_frame(1, {"cam1": [view("cam1", 1, emb=(0.0, 1.0))]})
        assert changed[("cam1", 1)] == 2  # new global, not the old one

    def test_max_age_prunes_stale_tracks(self) -> None:
        a = _assigner(max_age=5)
        a.assign_frame(0, {"cam1": [view("cam1", 1)]})
        later = a.assign_frame(10, {"cam1": [view("cam1", 1, emb=(0.0, 1.0))]})
        assert later[("cam1", 1)] == 2

    def test_class_mismatch_blocks_merge(self) -> None:
        a = _assigner()
        mapping = a.assign_frame(0, {"cam1": [view("cam1", 1, cls=0)], "cam2": [view("cam2", 1, cls=1)]})
        assert mapping[("cam1", 1)] != mapping[("cam2", 1)]

    def test_caller_supplied_cam_order_is_honored(self) -> None:
        a = _assigner()
        v1, v2 = view("cam1", 1, emb=(1.0, 0.0)), view("cam2", 1, emb=(1.0, 0.0))
        mapping = a.assign_frame(0, {"cam2": [v2], "cam1": [v1]}, cam_order=["cam1", "cam2"])
        assert mapping[("cam1", 1)] == mapping[("cam2", 1)] == 1

    def test_empty_frame_is_fine(self) -> None:
        a = _assigner()
        assert a.assign_frame(0, {"cam1": []}) == {}


class TestRecorderSeam:
    def test_recorder_receives_readme_documented_payload(self) -> None:
        a = _assigner()
        rec = FakeRecorder()
        v = view("cam1", 1)
        a.assign_frame(0, {"cam1": [v]}, recorder=rec)
        assert len(rec.calls) == 1
        frame_idx, views, payload = rec.calls[0]
        assert frame_idx == 0
        assert views == {"cam1": [v]}
        assert payload["frame"] == 0
        assert payload["threshold"] == pytest.approx(0.75)
        assert payload["max_age"] == 30
        assert isinstance(payload["steps"], list)

    def test_no_recorder_means_no_payload_work(self) -> None:
        a = _assigner()
        # must not raise, must still return the mapping
        assert a.assign_frame(0, {"cam1": [view("cam1", 1)]}) == {("cam1", 1): 1}

    def test_ensure_and_ensure_view_are_gone(self) -> None:
        # The fallback protocol that could mint state-less gids is deleted,
        # not deprecated: there is no second way in.
        a = _assigner()
        assert not hasattr(a, "ensure")
        assert not hasattr(a, "ensure_view")
        assert not hasattr(a, "_map_last_seen")
