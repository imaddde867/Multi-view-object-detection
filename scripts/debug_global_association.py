from __future__ import annotations

import numpy as np

from multiview.tracking.global_association import GlobalIDAssigner, GlobalTrackView


def _norm(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False).reshape(-1)
    norm = float(np.linalg.norm(vec) + 1e-9)
    return vec / norm


def _view(cam: str, local_id: int, cls_id: int, emb: np.ndarray) -> GlobalTrackView:
    return GlobalTrackView(
        cam=cam,
        local_id=local_id,
        cls_id=cls_id,
        xyxy=(0.0, 0.0, 10.0, 10.0),
        embedding=_norm(emb),
    )


def main() -> None:
    gid = GlobalIDAssigner(match_threshold=0.9, max_age=10)
    emb_a = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    emb_b = _norm(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    emb_c = _norm(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

    views_frame0 = {
        "cam1": [_view("cam1", 1, 0, emb_a), _view("cam1", 2, 0, emb_b)],
        "cam2": [_view("cam2", 1, 0, emb_a), _view("cam2", 2, 0, emb_b), _view("cam2", 3, 0, emb_c)],
    }
    gid.assign_frame(0, views_frame0, cam_order=["cam1", "cam2"])

    gid_a1 = gid.get("cam1", 1)
    gid_a2 = gid.get("cam2", 1)
    gid_b1 = gid.get("cam1", 2)
    gid_b2 = gid.get("cam2", 2)
    gid_c2 = gid.get("cam2", 3)

    assert gid_a1 == gid_a2, "Object A should share a global ID across cameras."
    assert gid_b1 == gid_b2, "Object B should share a global ID across cameras."
    assert gid_a1 != gid_b1, "Distinct objects should not share the same global ID."
    assert gid_c2 not in (gid_a1, gid_b1), "Extra objects should not steal a shared global ID."

    views_frame1 = {
        "cam1": [_view("cam1", 3, 0, emb_a)],
        "cam2": [_view("cam2", 1, 0, emb_a)],
    }
    gid.assign_frame(1, views_frame1, cam_order=["cam1", "cam2"])

    assert gid.get("cam1", 3) == gid_a1, "Re-ID should map to the existing global ID."

    emb_d1 = _norm(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    emb_d2 = _norm(np.array([0.0, 0.0, 0.6, 0.8], dtype=np.float32))

    views_frame2 = {
        "cam1": [_view("cam1", 4, 0, emb_d1)],
        "cam2": [_view("cam2", 4, 0, emb_d2)],
    }
    gid.assign_frame(2, views_frame2, cam_order=["cam1", "cam2"])

    gid_d1 = gid.get("cam1", 4)
    gid_d2 = gid.get("cam2", 4)
    assert gid_d1 != gid_d2, "Weak cross-view similarity should not force an early merge."

    views_frame3 = {
        "cam1": [_view("cam1", 4, 0, emb_d1)],
        "cam2": [_view("cam2", 4, 0, emb_d1)],
    }
    gid.assign_frame(3, views_frame3, cam_order=["cam1", "cam2"])

    assert gid.get("cam1", 4) == gid.get("cam2", 4), "Later matches should reconcile global IDs."

    print("OK: global association sanity checks passed.")


if __name__ == "__main__":
    main()
