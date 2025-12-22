from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from multiview.tracking.embeddings import cosine_sim


@dataclass(frozen=True)
class GlobalTrackView:
    cam: str
    local_id: int
    cls_id: int
    xyxy: tuple[float, float, float, float]
    embedding: np.ndarray
    world_xy: tuple[float, float] | None = None


@dataclass
class GlobalTrackState:
    global_id: int
    cls_id: int
    embedding: np.ndarray
    last_seen: int
    bboxes: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    world_xy: tuple[float, float] | None = None


class GlobalIDAssigner:
    def __init__(
        self,
        *,
        match_threshold: float = 0.75,
        max_age: int = 30,
        spatial_weight: float = 0.0,
        spatial_sigma: float = 1.0,
        spatial_max_dist: float = 0.0,
        embedding_alpha: float = 1.0,
        world_alpha: float = 1.0,
    ):
        self.match_threshold = float(match_threshold)
        self.max_age = int(max_age)
        self.spatial_weight = max(0.0, min(1.0, float(spatial_weight)))
        self.spatial_sigma = max(1e-6, float(spatial_sigma))
        self.spatial_max_dist = float(spatial_max_dist)
        self.embedding_alpha = max(0.0, min(1.0, float(embedding_alpha)))
        self.world_alpha = max(0.0, min(1.0, float(world_alpha)))
        self._next_gid = 1
        self._map: dict[tuple[str, int], int] = {}
        self._map_last_seen: dict[tuple[str, int], int] = {}
        self._global: dict[int, GlobalTrackState] = {}

    def get(self, cam: str, local_id: int) -> int | None:
        return self._map.get((cam, local_id))

    def ensure(self, cam: str, local_id: int) -> int:
        key = (cam, local_id)
        gid = self._map.get(key)
        if gid is None:
            gid = self._next_gid
            self._next_gid += 1
            self._map[key] = gid
        return gid

    def ensure_view(self, view: GlobalTrackView, frame_idx: int) -> int:
        key = (view.cam, view.local_id)
        gid = self._map.get(key)
        if gid is None or gid not in self._global:
            return self._new_global(view, frame_idx)
        self._map_last_seen[key] = frame_idx
        self._update_state(gid, view, frame_idx)
        return gid

    def unify(self, a: tuple[str, int], b: tuple[str, int]) -> int:
        ga = self._map.get(a)
        gb = self._map.get(b)
        if ga is None and gb is None:
            gid = self._next_gid
            self._next_gid += 1
            self._map[a] = gid
            self._map[b] = gid
            return gid
        if ga is None:
            self._map[a] = gb
            return gb  # type: ignore[return-value]
        if gb is None:
            self._map[b] = ga
            return ga
        if ga == gb:
            return ga
        keep = min(ga, gb)
        drop = max(ga, gb)
        for k, v in list(self._map.items()):
            if v == drop:
                self._map[k] = keep
        state_keep = self._global.get(keep)
        state_drop = self._global.get(drop)
        if state_drop is not None:
            if state_keep is None:
                self._global[keep] = state_drop
            else:
                if state_drop.last_seen > state_keep.last_seen:
                    state_keep.embedding = state_drop.embedding
                    state_keep.last_seen = state_drop.last_seen
                    state_keep.cls_id = state_drop.cls_id
                    if state_drop.world_xy is not None:
                        state_keep.world_xy = state_drop.world_xy
                elif state_keep.world_xy is None and state_drop.world_xy is not None:
                    state_keep.world_xy = state_drop.world_xy
                state_keep.bboxes.update(state_drop.bboxes)
            self._global.pop(drop, None)
        return keep

    def assign_frame(
        self,
        frame_idx: int,
        views_by_cam: dict[str, list[GlobalTrackView]],
        *,
        cam_order: list[str] | None = None,
        debug: bool = False,
    ) -> dict[str, Any] | None:
        self._prune(frame_idx)
        if cam_order is None:
            cam_order = list(views_by_cam.keys())

        debug_info: dict[str, Any] | None = None
        if debug:
            debug_info = {
                "frame": int(frame_idx),
                "threshold": float(self.match_threshold),
                "max_age": int(self.max_age),
                "steps": [],
            }

        assigned_by_cam: dict[str, set[int]] = {cam: set() for cam in cam_order}
        unassigned_by_cam: dict[str, list[GlobalTrackView]] = {cam: [] for cam in cam_order}
        assigned_views_by_cam: dict[str, list[GlobalTrackView]] = {cam: [] for cam in cam_order}

        for cam in cam_order:
            for view in views_by_cam.get(cam, []):
                key = (cam, view.local_id)
                gid = self._map.get(key)
                if gid is None or gid not in self._global:
                    unassigned_by_cam[cam].append(view)
                    continue
                state = self._global[gid]
                cls_match = view.cls_id == state.cls_id
                sim = cosine_sim(view.embedding, state.embedding)
                if not cls_match or sim < self.match_threshold:
                    self._map.pop(key, None)
                    self._map_last_seen.pop(key, None)
                    unassigned_by_cam[cam].append(view)
                    continue
                assigned_by_cam[cam].add(gid)
                assigned_views_by_cam[cam].append(view)
                self._map_last_seen[key] = frame_idx
                self._update_state(gid, view, frame_idx)

        for anchor_idx, anchor_cam in enumerate(cam_order):
            anchor_views = assigned_views_by_cam[anchor_cam]
            if not anchor_views:
                continue
            for other_cam in cam_order[anchor_idx + 1 :]:
                other_views = assigned_views_by_cam[other_cam]
                if not other_views:
                    continue
                cost = self._build_cost(anchor_views, other_views)
                row_ind, col_ind = linear_sum_assignment(cost)
                step = None
                if debug_info is not None:
                    step = {
                        "type": "merge_assigned",
                        "anchor_cam": anchor_cam,
                        "other_cam": other_cam,
                        "threshold": float(self.match_threshold),
                        "rows": [self._view_debug(v) for v in anchor_views],
                        "cols": [self._view_debug(v) for v in other_views],
                        "cost_matrix": self._cost_to_list(cost),
                        "assignments": [],
                    }

                for i, j in zip(row_ind.tolist(), col_ind.tolist()):
                    a = anchor_views[i]
                    b = other_views[j]
                    sim = 1.0 - float(cost[i, j])
                    gid_a = self._map.get((anchor_cam, a.local_id))
                    gid_b = self._map.get((other_cam, b.local_id))
                    reason = "ok"
                    merged = False
                    if sim < self.match_threshold:
                        reason = "below_threshold"
                    elif gid_a is None or gid_b is None:
                        reason = "missing_gid"
                    elif gid_a == gid_b:
                        reason = "same_gid"
                    else:
                        keep = min(gid_a, gid_b)
                        drop = max(gid_a, gid_b)
                        conflict = any(keep in gids and drop in gids for gids in assigned_by_cam.values())
                        if conflict:
                            reason = "conflict"
                        else:
                            merged = True
                            self.unify((anchor_cam, a.local_id), (other_cam, b.local_id))
                            for gids in assigned_by_cam.values():
                                if drop in gids:
                                    gids.discard(drop)
                                    gids.add(keep)
                    if step is not None:
                        step["assignments"].append(
                            {
                                "row": int(i),
                                "col": int(j),
                                "sim": float(sim),
                                "merged": bool(merged),
                                "reason": reason,
                            }
                        )

                if step is not None:
                    debug_info["steps"].append(step)

        for cam in cam_order:
            views = unassigned_by_cam[cam]
            if not views:
                continue
            candidates = [s for s in self._global.values() if s.global_id not in assigned_by_cam[cam]]
            if not candidates:
                continue

            cost = self._build_cost(views, candidates)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_rows: set[int] = set()
            step: dict[str, Any] | None = None
            if debug_info is not None:
                step = {
                    "type": "match_existing",
                    "camera": cam,
                    "threshold": float(self.match_threshold),
                    "rows": [self._view_debug(v) for v in views],
                    "cols": [self._global_debug(s) for s in candidates],
                    "cost_matrix": self._cost_to_list(cost),
                    "assignments": [],
                }

            for i, j in zip(row_ind.tolist(), col_ind.tolist()):
                view = views[i]
                state = candidates[j]
                sim = 1.0 - float(cost[i, j])
                cls_match = view.cls_id == state.cls_id
                accepted = cls_match and sim >= self.match_threshold
                reason = "ok" if accepted else ("class_mismatch" if not cls_match else "below_threshold")
                if step is not None:
                    step["assignments"].append(
                        {
                            "row": int(i),
                            "col": int(j),
                            "sim": float(sim),
                            "accepted": bool(accepted),
                            "reason": reason,
                        }
                    )
                if not accepted:
                    continue
                gid = state.global_id
                key = (cam, view.local_id)
                self._map[key] = gid
                self._map_last_seen[key] = frame_idx
                self._update_state(gid, view, frame_idx)
                assigned_by_cam[cam].add(gid)
                matched_rows.add(i)

            if step is not None:
                debug_info["steps"].append(step)

            unassigned_by_cam[cam] = [v for idx, v in enumerate(views) if idx not in matched_rows]

        for anchor_idx, anchor_cam in enumerate(cam_order):
            anchor_views = unassigned_by_cam[anchor_cam]
            if not anchor_views:
                continue
            for other_cam in cam_order[anchor_idx + 1 :]:
                other_views = unassigned_by_cam[other_cam]
                if not other_views:
                    continue

                def gate_fn(a: GlobalTrackView, _b: GlobalTrackView) -> bool:
                    gid = self._map.get((anchor_cam, a.local_id))
                    if gid is None:
                        return True
                    return gid not in assigned_by_cam[other_cam]

                cost = self._build_cost(anchor_views, other_views, gate_fn=gate_fn)
                row_ind, col_ind = linear_sum_assignment(cost)
                matched_other: set[int] = set()
                step = None
                if debug_info is not None:
                    step = {
                        "type": "match_new",
                        "anchor_cam": anchor_cam,
                        "other_cam": other_cam,
                        "threshold": float(self.match_threshold),
                        "rows": [self._view_debug(v) for v in anchor_views],
                        "cols": [self._view_debug(v) for v in other_views],
                        "cost_matrix": self._cost_to_list(cost),
                        "assignments": [],
                    }

                for i, j in zip(row_ind.tolist(), col_ind.tolist()):
                    a = anchor_views[i]
                    b = other_views[j]
                    sim = 1.0 - float(cost[i, j])
                    cls_match = a.cls_id == b.cls_id
                    gate_ok = gate_fn(a, b)
                    accepted = cls_match and gate_ok and sim >= self.match_threshold
                    reason = "ok"
                    if not cls_match:
                        reason = "class_mismatch"
                    elif not gate_ok:
                        reason = "gid_in_use"
                    elif sim < self.match_threshold:
                        reason = "below_threshold"
                    if step is not None:
                        step["assignments"].append(
                            {
                                "row": int(i),
                                "col": int(j),
                                "sim": float(sim),
                                "accepted": bool(accepted),
                                "reason": reason,
                            }
                        )
                    if not accepted:
                        continue
                    key_a = (anchor_cam, a.local_id)
                    gid = self._map.get(key_a)
                    if gid is None or gid not in self._global:
                        gid = self._new_global(a, frame_idx)
                    else:
                        self._map_last_seen[key_a] = frame_idx
                        self._update_state(gid, a, frame_idx)
                    assigned_by_cam[anchor_cam].add(gid)

                    key_b = (other_cam, b.local_id)
                    self._map[key_b] = gid
                    self._map_last_seen[key_b] = frame_idx
                    self._update_state(gid, b, frame_idx)
                    assigned_by_cam[other_cam].add(gid)
                    matched_other.add(j)

                if step is not None:
                    debug_info["steps"].append(step)
                unassigned_by_cam[other_cam] = [v for idx, v in enumerate(other_views) if idx not in matched_other]

        for cam in cam_order:
            for view in unassigned_by_cam[cam]:
                key = (cam, view.local_id)
                if key in self._map and self._map.get(key) in self._global:
                    continue
                gid = self._new_global(view, frame_idx)
                assigned_by_cam[cam].add(gid)

        return debug_info

    def associate_anchor(
        self,
        anchor_cam: str,
        anchor_tracks: list[GlobalTrackView],
        other_cam: str,
        other_tracks: list[GlobalTrackView],
    ) -> None:
        # Legacy anchor association (does not update the global registry state).
        if not anchor_tracks or not other_tracks:
            return

        cost = np.ones((len(anchor_tracks), len(other_tracks)), dtype=np.float32) * 1e3
        for i, ta in enumerate(anchor_tracks):
            for j, tb in enumerate(other_tracks):
                if ta.cls_id != tb.cls_id:
                    continue
                sim = cosine_sim(ta.embedding, tb.embedding)
                cost[i, j] = 1.0 - sim

        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            sim = 1.0 - float(cost[i, j])
            if sim < self.match_threshold:
                continue
            self.unify((anchor_cam, anchor_tracks[i].local_id), (other_cam, other_tracks[j].local_id))

    def _new_global(self, view: GlobalTrackView, frame_idx: int) -> int:
        gid = self._next_gid
        self._next_gid += 1
        key = (view.cam, view.local_id)
        self._map[key] = gid
        self._map_last_seen[key] = frame_idx
        self._global[gid] = GlobalTrackState(
            global_id=gid,
            cls_id=view.cls_id,
            embedding=view.embedding.copy(),
            last_seen=frame_idx,
            bboxes={view.cam: view.xyxy},
            world_xy=view.world_xy,
        )
        return gid

    def _update_state(self, gid: int, view: GlobalTrackView, frame_idx: int) -> None:
        state = self._global.get(gid)
        if state is None:
            self._global[gid] = GlobalTrackState(
                global_id=gid,
                cls_id=view.cls_id,
                embedding=view.embedding.copy(),
                last_seen=frame_idx,
                bboxes={view.cam: view.xyxy},
                world_xy=view.world_xy,
            )
            return
        state.cls_id = view.cls_id
        state.embedding = self._blend_embedding(state.embedding, view.embedding)
        state.last_seen = frame_idx
        state.bboxes[view.cam] = view.xyxy
        if view.world_xy is not None:
            state.world_xy = self._blend_world(state.world_xy, view.world_xy)

    def _blend_embedding(self, prev: np.ndarray, new: np.ndarray) -> np.ndarray:
        alpha = self.embedding_alpha
        if alpha >= 1.0:
            blended = new.astype(np.float32, copy=True)
        elif alpha <= 0.0:
            blended = prev.astype(np.float32, copy=True)
        else:
            blended = prev.astype(np.float32, copy=False) * (1.0 - alpha) + new.astype(np.float32, copy=False) * alpha
        norm = float(np.linalg.norm(blended) + 1e-9)
        return blended / norm

    def _blend_world(
        self, prev: tuple[float, float] | None, new: tuple[float, float]
    ) -> tuple[float, float]:
        if prev is None:
            return (float(new[0]), float(new[1]))
        alpha = self.world_alpha
        if alpha >= 1.0:
            return (float(new[0]), float(new[1]))
        if alpha <= 0.0:
            return (float(prev[0]), float(prev[1]))
        return (
            float(prev[0]) * (1.0 - alpha) + float(new[0]) * alpha,
            float(prev[1]) * (1.0 - alpha) + float(new[1]) * alpha,
        )

    def _prune(self, frame_idx: int) -> None:
        if self.max_age <= 0:
            return
        stale_gids = [gid for gid, st in self._global.items() if frame_idx - st.last_seen > self.max_age]
        for gid in stale_gids:
            self._global.pop(gid, None)
        for key, last_seen in list(self._map_last_seen.items()):
            gid = self._map.get(key)
            if gid is None:
                self._map_last_seen.pop(key, None)
                continue
            if frame_idx - last_seen > self.max_age or gid not in self._global:
                self._map.pop(key, None)
                self._map_last_seen.pop(key, None)

    def _build_cost(
        self,
        rows: list[GlobalTrackView],
        cols: list[Any],
        *,
        gate_fn: Callable[[GlobalTrackView, Any], bool] | None = None,
    ) -> np.ndarray:
        cost = np.ones((len(rows), len(cols)), dtype=np.float32) * 1e3
        for i, ra in enumerate(rows):
            for j, cb in enumerate(cols):
                if ra.cls_id != cb.cls_id:
                    continue
                if gate_fn is not None and not gate_fn(ra, cb):
                    continue
                sim = cosine_sim(ra.embedding, cb.embedding)
                spatial_weight = self.spatial_weight
                if spatial_weight > 0.0:
                    ra_world = getattr(ra, "world_xy", None)
                    cb_world = getattr(cb, "world_xy", None)
                    if ra_world is not None and cb_world is not None:
                        dx = float(ra_world[0]) - float(cb_world[0])
                        dy = float(ra_world[1]) - float(cb_world[1])
                        dist = float(np.hypot(dx, dy))
                        if self.spatial_max_dist > 0.0 and dist > self.spatial_max_dist:
                            continue
                        spatial_sim = float(np.exp(-(dist * dist) / (2.0 * self.spatial_sigma * self.spatial_sigma)))
                        sim = (1.0 - spatial_weight) * sim + spatial_weight * spatial_sim
                cost[i, j] = 1.0 - sim
        return cost

    @staticmethod
    def _view_debug(view: GlobalTrackView) -> dict[str, Any]:
        payload = {
            "cam": view.cam,
            "local_id": int(view.local_id),
            "cls_id": int(view.cls_id),
            "bbox_xyxy": [float(v) for v in view.xyxy],
        }
        if view.world_xy is not None:
            payload["world_xy"] = [float(v) for v in view.world_xy]
        return payload

    @staticmethod
    def _global_debug(state: GlobalTrackState) -> dict[str, Any]:
        payload = {
            "global_id": int(state.global_id),
            "cls_id": int(state.cls_id),
            "last_seen": int(state.last_seen),
        }
        if state.world_xy is not None:
            payload["world_xy"] = [float(v) for v in state.world_xy]
        return payload

    @staticmethod
    def _cost_to_list(cost: np.ndarray) -> list[list[float]]:
        return [[float(v) for v in row.tolist()] for row in cost]
