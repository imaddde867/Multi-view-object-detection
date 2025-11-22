"""Run multi-view object detection using YOLO outputs + triangulation."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List

import numpy as np

from multiview_matching import match_and_triangulate
from triangulation import Triangulator
from yolo_loader import YoloDetectionLoader, load_class_names


def parse_camera_indices(camera_arg: str) -> List[int]:
    tokens = [tok.strip() for tok in camera_arg.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("At least one camera index is required")
    return [int(tok) for tok in tokens]


def load_calibration(file_path: str) -> List[np.ndarray]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Calibration file '{file_path}' not found. Run create_calibration.py first."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    projections: Dict[int, np.ndarray] = {}
    for cam_id, cam_data in data.get("cameras", {}).items():
        P = np.array(cam_data["P"], dtype=np.float64)
        idx = int(cam_id.replace("camera", ""))
        projections[idx] = P

    if not projections:
        raise RuntimeError("Calibration file does not contain any cameras")

    return [projections[i] for i in sorted(projections.keys())]


def gather_frames(loader: YoloDetectionLoader, args) -> List[int]:
    if args.all_frames:
        frames = loader.available_frames()
        if args.max_frames:
            frames = frames[: args.max_frames]
        return frames
    return [args.frame]


def save_point_cloud(points: Iterable[np.ndarray], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-view object detection pipeline")
    parser.add_argument("--calibration", default="epfl-calibration.json")
    parser.add_argument("--labels-dir", default="Yolov8_Multi-view-main/yolo_labels")
    parser.add_argument("--image-root", default="multiclass_ground_truth_images")
    parser.add_argument("--data-yaml", default="Yolov8_Multi-view-main/data.yaml")
    parser.add_argument("--frame", type=int, default=150, help="Frame index to process")
    parser.add_argument("--all-frames", action="store_true", help="Process every available frame")
    parser.add_argument("--max-frames", type=int, help="Optional cap when using --all-frames")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--max-reproj-error", type=float, default=10.0)
    parser.add_argument("--pixel-tolerance", type=float, default=15.0)
    parser.add_argument("--merge-distance", type=float, default=500.0)
    parser.add_argument(
        "--cameras",
        default="0,1,2,3,4,5",
        help="Comma-separated camera indices to consider",
    )
    parser.add_argument(
        "--output-obj",
        default="output_cloud.obj",
        help="Where to save aggregated 3D points (set blank to skip)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    camera_indices = parse_camera_indices(args.cameras)
    projections = load_calibration(args.calibration)
    triangulator = Triangulator(projections)

    class_names = load_class_names(args.data_yaml)

    print("=" * 70)
    print("EPFL Multi-View Object Detection")
    print("=" * 70)
    print(f"Calibration: {args.calibration}")
    print(f"Labels dir : {args.labels_dir}")
    print(f"Images dir : {args.image_root}")
    print(f"Cameras    : {camera_indices}")

    loader = YoloDetectionLoader(
        labels_dir=args.labels_dir,
        image_root=args.image_root,
        class_names=class_names,
        camera_indices=camera_indices,
    )

    frames = gather_frames(loader, args)
    if not frames:
        print("No frames found. Exiting.")
        return

    print(f"Frames     : {frames[:5]}{'...' if len(frames) > 5 else ''} (total {len(frames)})")

    aggregated_points: List[np.ndarray] = []

    for frame_id in frames:
        detections = loader.load_frame(frame_id)
        detections_count = sum(len(v) for v in detections.values())
        print(f"\nFrame {frame_id}: {detections_count} detections across {len(detections)} cameras")

        if len(detections) < 2:
            print("  ↳ Skipped (need detections from at least two cameras)")
            continue

        objects, stats = match_and_triangulate(
            detections,
            triangulator,
            frame_id,
            min_confidence=args.min_conf,
            max_reprojection_error=args.max_reproj_error,
            pixel_tolerance=args.pixel_tolerance,
            merge_distance=args.merge_distance,
        )

        if not objects:
            print(
                f"  ↳ No valid matches (pairs tried: {stats.pair_candidates}, accepted: {stats.accepted_pairs})"
            )
            continue

        avg_views = sum(obj.num_views() for obj in objects) / len(objects)
        print(
            f"  ↳ {len(objects)} objects | {stats.accepted_pairs}/{stats.pair_candidates} valid pairs | "
            f"avg views {avg_views:.1f}"
        )

        for obj in objects:
            aggregated_points.append(obj.point_3d)

    if aggregated_points and args.output_obj:
        save_point_cloud(aggregated_points, args.output_obj)
        print(f"\nSaved {len(aggregated_points)} 3D points to {args.output_obj}")

    print("\nDone.")


if __name__ == "__main__":
    main()
