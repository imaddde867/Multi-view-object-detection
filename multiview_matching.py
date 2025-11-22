"""Cross-view matching utilities for YOLO detections."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from triangulation import Triangulator
from yolo_loader import YoloDetection


@dataclass
class MatchStats:
    frame_id: int
    pair_candidates: int = 0
    accepted_pairs: int = 0
    merged_objects: int = 0


@dataclass
class TriangulatedObject:
    frame_id: int
    class_id: int
    class_name: str
    point_3d: np.ndarray
    views: Dict[int, YoloDetection] = field(default_factory=dict)
    reprojection_errors: Dict[int, float] = field(default_factory=dict)

    def num_views(self) -> int:
        return len(self.views)


def match_and_triangulate(
    detections_by_cam: Dict[int, Sequence[YoloDetection]],
    triangulator: Triangulator,
    frame_id: int,
    *,
    min_confidence: float = 0.1,
    max_reprojection_error: float = 10.0,
    pixel_tolerance: float = 20.0,
    merge_distance: float = 400.0,
) -> Tuple[List[TriangulatedObject], MatchStats]:
    """Generate cross-view matches and triangulated 3D objects."""

    cameras = sorted(detections_by_cam.keys())
    stats = MatchStats(frame_id=frame_id)
    candidates: List[TriangulatedObject] = []

    if len(cameras) < 2:
        return [], stats

    for i, cam_a in enumerate(cameras):
        dets_a = [d for d in detections_by_cam[cam_a] if d.confidence >= min_confidence]
        if not dets_a:
            continue

        for cam_b in cameras[i + 1 :]:
            dets_b = [d for d in detections_by_cam[cam_b] if d.confidence >= min_confidence]
            if not dets_b:
                continue

            for det_a in dets_a:
                for det_b in dets_b:
                    if det_a.class_id != det_b.class_id:
                        continue

                    stats.pair_candidates += 1
                    triangulated = triangulator.triangulate_points(
                        cam_a, cam_b, [det_a.center], [det_b.center]
                    )
                    if len(triangulated) == 0:
                        continue

                    point = triangulated[0]
                    error_a = _reprojection_error(triangulator, point, cam_a, det_a)
                    error_b = _reprojection_error(triangulator, point, cam_b, det_b)

                    if error_a > max_reprojection_error or error_b > max_reprojection_error:
                        continue

                    stats.accepted_pairs += 1

                    candidate = TriangulatedObject(
                        frame_id=frame_id,
                        class_id=det_a.class_id,
                        class_name=det_a.class_name,
                        point_3d=point,
                        views={cam_a: det_a, cam_b: det_b},
                        reprojection_errors={cam_a: error_a, cam_b: error_b},
                    )

                    _attach_supporting_views(
                        candidate,
                        detections_by_cam,
                        triangulator,
                        pixel_tolerance,
                    )
                    candidates.append(candidate)

    merged = _merge_candidates(candidates, merge_distance)
    stats.merged_objects = len(merged)
    return merged, stats


def _attach_supporting_views(
    triangulated: TriangulatedObject,
    detections_by_cam: Dict[int, Sequence[YoloDetection]],
    triangulator: Triangulator,
    pixel_tolerance: float,
) -> None:
    for cam_idx, detections in detections_by_cam.items():
        if cam_idx in triangulated.views:
            continue

        projected = triangulator.project_point(triangulated.point_3d, cam_idx)
        if projected is None:
            continue

        best_det = None
        best_error = float("inf")

        for det in detections:
            if det.class_id != triangulated.class_id:
                continue

            error = np.linalg.norm(np.array(det.center) - projected)
            if error < best_error:
                best_error = error
                best_det = det

        if best_det and best_error <= pixel_tolerance:
            triangulated.views[cam_idx] = best_det
            triangulated.reprojection_errors[cam_idx] = best_error


def _merge_candidates(
    candidates: Sequence[TriangulatedObject], merge_distance: float
) -> List[TriangulatedObject]:
    merged: List[TriangulatedObject] = []

    for candidate in candidates:
        merged_into = None
        for existing in merged:
            if candidate.class_id != existing.class_id:
                continue

            if np.linalg.norm(existing.point_3d - candidate.point_3d) <= merge_distance:
                merged_into = existing
                break

        if merged_into is None:
            merged.append(candidate)
            continue

        total_views = merged_into.num_views() + candidate.num_views()
        merged_into.point_3d = (
            (merged_into.point_3d * merged_into.num_views())
            + (candidate.point_3d * candidate.num_views())
        ) / max(total_views, 1)

        for cam_idx, det in candidate.views.items():
            if cam_idx not in merged_into.views or det.confidence > merged_into.views[cam_idx].confidence:
                merged_into.views[cam_idx] = det
                merged_into.reprojection_errors[cam_idx] = candidate.reprojection_errors.get(
                    cam_idx, merged_into.reprojection_errors.get(cam_idx, 0.0)
                )

    return merged


def _reprojection_error(
    triangulator: Triangulator,
    point_3d: np.ndarray,
    cam_idx: int,
    detection: YoloDetection,
) -> float:
    projected = triangulator.project_point(point_3d, cam_idx)
    if projected is None:
        return float("inf")
    return float(np.linalg.norm(np.array(detection.center) - projected))

