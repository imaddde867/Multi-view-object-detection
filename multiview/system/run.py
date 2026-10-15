from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from multiview.config import merge
from multiview.detector import YoloDetector, parse_kv_map
from multiview.frames import iter_group_frames
from multiview.tracking.global_association import GlobalIDAssigner, GlobalTrackView
from multiview.tracking.simple_tracker import SimpleTracker
from multiview.utils.video import VideoInfo, open_video


def _default_cfg() -> dict[str, Any]:
    return {
        "cameras": {},
        "groups": {},
        "detector": {
            "model": "",
            "imgsz": 960,
            "conf": 0.35,
            "iou": 0.5,
            "max_det": 0,
            "targets": "person,car",
            "source_map": "truck=car,motorcycle=car,bicycle=car",
        },
        "tracker": {
            "max_age": 30,
            "match_threshold": 0.5,
            "global_match_threshold": 0.75,
            "global_max_age": 30,
            "spatial_weight": 0.0,
            "spatial_sigma": 1.0,
            "spatial_max_dist": 0.0,
            "embedding_alpha": 1.0,
            "world_alpha": 1.0,
            "embedder": {"type": "colorhist", "bins": 16},
        },
        "run": {"frame_stride": 1, "max_frames": None, "groups": []},
        "output": {"dir": "results/system", "write_video": False, "video_fps": 0.0},
        "runtime": {"device": "", "half": False},
        "debug": {"global_assoc": False, "log_path": ""},
    }


def _draw_tracks(frame_bgr: np.ndarray, tracks: list[dict[str, Any]], class_names: list[str], color_seed: int) -> np.ndarray:
    out = frame_bgr.copy()
    for tr in tracks:
        gid = int(tr["global_id"])
        local = int(tr["local_id"])
        cls_id = int(tr["cls_id"])
        x0, y0, x1, y1 = [int(v) for v in tr["bbox_xyxy"]]
        rng = np.random.default_rng(color_seed + gid * 97)
        color = tuple(int(x) for x in rng.integers(20, 235, size=3))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"G{gid} L{local} {class_names[cls_id]}"
        cv2.putText(out, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def _as_bool(value: Any) -> bool:
    """Coerce YAML/CLI truthy values; bool("false") is a trap we refuse."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "1", "on"}:
            return True
        if low in {"false", "no", "0", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")


def _as_targets(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    return [t.strip() for t in str(value).split(",") if t.strip()]


def _as_optional_int(value: Any) -> int | None:
    if value in (None, "", 0):
        return None
    return int(value)


def _camera_homography(entry: dict[str, Any], cam: str) -> np.ndarray | None:
    homography = entry.get("homography")
    if homography is None:
        image_points = entry.get("image_points")
        world_points = entry.get("world_points")
        if image_points is None or world_points is None:
            return None
        img = np.asarray(image_points, dtype=np.float32)
        world = np.asarray(world_points, dtype=np.float32)
        if img.shape != world.shape or img.ndim != 2 or img.shape[0] < 4 or img.shape[1] != 2:
            raise ValueError(
                f"Camera {cam} image_points/world_points must be Nx2 with N>=4 and matching shapes."
            )
        mat, _ = cv2.findHomography(img, world, method=0)
        if mat is None:
            raise ValueError(f"Camera {cam} homography could not be estimated from points.")
        return mat.astype(np.float32)
    mat = np.asarray(homography, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Camera {cam} homography must be a 3x3 matrix.")
    return mat.astype(np.float32)


@dataclass(frozen=True)
class RunConfig:
    """Typed view of a merged run config, built once at the run seam.

    The run pipeline used to cast every value from the raw dict at its use
    site; this dataclass performs each cast and validation once, with clear
    errors, and gives the run loop typed fields to read.
    """

    cameras: dict[str, Any]
    groups: dict[str, Any]
    cam_homographies: dict[str, np.ndarray]
    model: str
    imgsz: int
    conf: float
    iou: float
    max_det: int
    targets: list[str]
    source_map: dict[str, str]
    max_age: int
    match_threshold: float
    global_match_threshold: float
    global_max_age: int
    spatial_weight: float
    spatial_sigma: float
    spatial_max_dist: float
    embedding_alpha: float
    world_alpha: float
    embedder_type: str
    embedder_bins: int
    embedder_device: str
    frame_stride: int
    max_frames: int | None
    run_groups: list[str]
    out_dir: Path
    write_video: bool
    video_fps: float
    device: str
    half: bool
    debug_assoc: bool
    debug_log_path: str

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RunConfig:
        cameras = cfg.get("cameras") or {}
        groups = cfg.get("groups") or {}
        if not cameras:
            raise ValueError("Config must define `cameras` and `groups`.")
        if not groups:
            raise ValueError("Config must define `cameras` and `groups`.")
        detector = cfg.get("detector", {})
        tracker = cfg.get("tracker", {})
        run_cfg = cfg.get("run", {})
        out_cfg = cfg.get("output", {})
        runtime = cfg.get("runtime", {})
        debug = cfg.get("debug", {})

        model = str(detector.get("model", "")).strip()
        if not model:
            raise ValueError("Missing detector.model in config (weights path).")

        embedder_cfg = tracker.get("embedder", {}) or {}
        if isinstance(embedder_cfg, str):
            embedder_cfg = {"type": embedder_cfg}
        embedder_type = str(embedder_cfg.get("type", "colorhist"))
        embedder_device = str(embedder_cfg.get("device", runtime.get("device", "cpu")) or "cpu")
        if embedder_device.isdigit():
            embedder_device = f"cuda:{embedder_device}"

        cam_homographies: dict[str, np.ndarray] = {}
        for cam, cam_entry in cameras.items():
            if not isinstance(cam_entry, dict):
                continue
            mat = _camera_homography(cam_entry, str(cam))
            if mat is not None:
                cam_homographies[str(cam)] = mat

        raw_map = detector.get("source_map", "")
        source_map = parse_kv_map(raw_map) if isinstance(raw_map, str) else dict(raw_map or {})

        return cls(
            cameras=cameras,
            groups=groups,
            cam_homographies=cam_homographies,
            model=model,
            imgsz=int(detector.get("imgsz", 960)),
            conf=float(detector.get("conf", 0.35)),
            iou=float(detector.get("iou", 0.5)),
            max_det=int(detector.get("max_det", 0)),
            targets=_as_targets(detector.get("targets", "person,car")),
            source_map=source_map,
            max_age=int(tracker.get("max_age", 30)),
            match_threshold=float(tracker.get("match_threshold", 0.5)),
            global_match_threshold=float(tracker.get("global_match_threshold", 0.75)),
            global_max_age=int(tracker.get("global_max_age", tracker.get("max_age", 30))),
            spatial_weight=float(tracker.get("spatial_weight", 0.0)),
            spatial_sigma=float(tracker.get("spatial_sigma", 1.0)),
            spatial_max_dist=float(tracker.get("spatial_max_dist", 0.0)),
            embedding_alpha=float(tracker.get("embedding_alpha", 1.0)),
            world_alpha=float(tracker.get("world_alpha", 1.0)),
            embedder_type=embedder_type,
            embedder_bins=int(embedder_cfg.get("bins", 16)),
            embedder_device=embedder_device,
            frame_stride=int(run_cfg.get("frame_stride", 1) or 1),
            max_frames=_as_optional_int(run_cfg.get("max_frames")),
            run_groups=[str(g) for g in run_cfg.get("groups", [])],
            out_dir=Path(str(out_cfg.get("dir", "results/system"))),
            write_video=_as_bool(out_cfg.get("write_video", False)),
            video_fps=float(out_cfg.get("video_fps") or 0.0),
            device=str(runtime.get("device", "")).strip(),
            half=_as_bool(runtime.get("half", False)),
            debug_assoc=_as_bool(debug.get("global_assoc", False)),
            debug_log_path=str(debug.get("log_path", "")).strip(),
        )


def run_multiview(cfg: dict[str, Any]) -> None:
    c = RunConfig.from_config(merge(_default_cfg(), cfg))

    detector = YoloDetector(c.model, device=c.device, half=c.half)
    target_to_id = {name: i for i, name in enumerate(c.targets)}

    if c.embedder_type == "torch_resnet18":
        from multiview.tracking.embeddings import TorchResNet18Embedder

        embedder = TorchResNet18Embedder(device=c.embedder_device)
    else:
        from multiview.tracking.embeddings import ColorHistEmbedder

        embedder = ColorHistEmbedder(bins=c.embedder_bins)

    selected_groups = [g for g in c.run_groups if g.strip()]
    if not selected_groups:
        selected_groups = [str(g) for g in c.groups.keys()]

    output_root = c.out_dir
    output_root.mkdir(parents=True, exist_ok=True)

    debug_assoc = c.debug_assoc
    debug_log_path = c.debug_log_path
    cameras_cfg = c.cameras
    groups_cfg = c.groups

    cam_homographies = c.cam_homographies

    def _world_xy(cam: str, xyxy: tuple[float, float, float, float]) -> tuple[float, float] | None:
        mat = cam_homographies.get(cam)
        if mat is None:
            return None
        x0, y0, x1, y1 = xyxy
        cx = 0.5 * (x0 + x1)
        cy = y1
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, mat)[0, 0]
        return (float(mapped[0]), float(mapped[1]))

    def _camera_info(info: VideoInfo) -> dict[str, Any]:
        return {
            "path": str(info.path),
            "width": int(info.width),
            "height": int(info.height),
            "fps": float(info.fps),
            "frame_count": int(info.frame_count),
        }

    def _estimate_fps(path: Path, *, sample_frames: int = 10) -> float:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return 0.0
        timestamps: list[float] = []
        for _ in range(sample_frames):
            ret, _frame = cap.read()
            if not ret:
                break
            timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
        cap.release()
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
        if not deltas:
            return 0.0
        median_delta = float(np.median(np.asarray(deltas, dtype=np.float32)))
        if median_delta <= 0.0:
            return 0.0
        return 1000.0 / median_delta

    for group_name in selected_groups:
        cam_names = groups_cfg.get(group_name)
        if not cam_names:
            raise ValueError(f"Group {group_name} not found in config.")
        cam_names = [str(c) for c in cam_names]

        caps: dict[str, cv2.VideoCapture] = {}
        infos: dict[str, VideoInfo] = {}
        writer: cv2.VideoWriter | None = None
        debug_fp = None
        try:
            for cam in cam_names:
                cam_entry = cameras_cfg.get(cam)
                if not cam_entry:
                    raise ValueError(f"Camera {cam} referenced in group {group_name} but not defined in cameras.")
                src = cam_entry["source"] if isinstance(cam_entry, dict) else str(cam_entry)
                cap, info = open_video(Path(src))
                caps[cam] = cap
                infos[cam] = info

            effective_fps: dict[str, float] = {}
            for cam in cam_names:
                fps = _estimate_fps(infos[cam].path)
                if fps > 0.0:
                    effective_fps[cam] = fps

            trackers = {
                cam: SimpleTracker(embedder, max_age=c.max_age, match_threshold=c.match_threshold)
                for cam in cam_names
            }
            gid = GlobalIDAssigner(
                match_threshold=c.global_match_threshold,
                max_age=c.global_max_age,
                spatial_weight=c.spatial_weight,
                spatial_sigma=c.spatial_sigma,
                spatial_max_dist=c.spatial_max_dist,
                embedding_alpha=c.embedding_alpha,
                world_alpha=c.world_alpha,
            )

            fps_candidates = [
                (c, effective_fps[c]) for c in cam_names if c in effective_fps and effective_fps[c] > 0.0
            ]
            if fps_candidates:
                sync_cam, sync_fps = min(fps_candidates, key=lambda x: x[1])
            else:
                meta_candidates = [(c, infos[c].fps) for c in cam_names if infos[c].fps > 0.0]
                sync_cam = min(meta_candidates, key=lambda x: x[1])[0] if meta_candidates else cam_names[0]
                sync_fps = float(infos[sync_cam].fps or 0.0)

            out_json: dict[str, Any] = {
                "metadata": {
                    "group": group_name,
                    "cameras": {c: _camera_info(infos[c]) for c in cam_names},
                    "classes": c.targets,
                    "model": c.model,
                },
                "frames": [],
            }

            write_video = c.write_video
            video_out_path = output_root / f"{group_name}.avi"
            color_seed = 1337
            if debug_assoc:
                debug_path = Path(debug_log_path) if debug_log_path else output_root / f"{group_name}_global_assoc.jsonl"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_fp = debug_path.open("w")

            for raw_idx, frames in iter_group_frames(
                caps,
                stride=c.frame_stride,
                max_frames=c.max_frames,
                align="timestamp",
                sync_cam=sync_cam,
            ):
                batch = [frames[cam] for cam in cam_names]
                dets_batch = detector.predict_dets(
                    batch,
                    conf=c.conf,
                    iou=c.iou,
                    imgsz=c.imgsz,
                    max_det=c.max_det,
                    source_map=c.source_map,
                    target_to_id=target_to_id,
                )

                per_cam_tracks: dict[str, list[dict[str, Any]]] = {}
                views_by_cam: dict[str, list[GlobalTrackView]] = {}
                views_by_key: dict[tuple[str, int], GlobalTrackView] = {}

                for cam, frame, mapped_dets in zip(cam_names, batch, dets_batch):
                    tracks = trackers[cam].update(frame, mapped_dets)
                    out_tracks: list[dict[str, Any]] = []
                    views: list[GlobalTrackView] = []
                    for t in tracks:
                        local = int(t.track_id)
                        view = GlobalTrackView(
                            cam=cam,
                            local_id=local,
                            cls_id=int(t.cls_id),
                            xyxy=t.xyxy,
                            embedding=t.embedding,
                            world_xy=_world_xy(cam, t.xyxy),
                        )
                        out_tracks.append(
                            {
                                "local_id": local,
                                "cls_id": int(t.cls_id),
                                "cls_name": c.targets[int(t.cls_id)],
                                "bbox_xyxy": [float(v) for v in t.xyxy],
                                "score": float(t.score),
                            }
                        )
                        views.append(view)
                        views_by_key[(cam, local)] = view
                    per_cam_tracks[cam] = out_tracks
                    views_by_cam[cam] = views

                frame_idx = raw_idx
                assoc_debug = gid.assign_frame(frame_idx, views_by_cam, cam_order=cam_names, debug=debug_assoc)

                for cam in cam_names:
                    for tr in per_cam_tracks[cam]:
                        local = int(tr["local_id"])
                        global_id = gid.get(cam, local)
                        if global_id is None:
                            view = views_by_key.get((cam, local))
                            if view is not None:
                                global_id = gid.ensure_view(view, frame_idx)
                            else:
                                global_id = gid.ensure(cam, local)
                        tr["global_id"] = global_id

                out_json["frames"].append({"frame": frame_idx, "cameras": per_cam_tracks})

                if debug_fp is not None:
                    tracks_log: dict[str, list[dict[str, Any]]] = {}
                    for cam in cam_names:
                        cam_tracks: list[dict[str, Any]] = []
                        for tr in per_cam_tracks[cam]:
                            key = (cam, int(tr["local_id"]))
                            view = views_by_key.get(key)
                            embedding = []
                            if view is not None:
                                embedding = view.embedding.astype(np.float32, copy=False).reshape(-1).tolist()
                            cam_tracks.append(
                                {
                                    "camera_id": cam,
                                    "local_id": int(tr["local_id"]),
                                    "global_id": int(tr["global_id"]),
                                    "bbox_xyxy": [float(v) for v in tr["bbox_xyxy"]],
                                    "score": float(tr["score"]),
                                    "cls_id": int(tr["cls_id"]),
                                    "cls_name": str(tr["cls_name"]),
                                    "embedding": embedding,
                                }
                            )
                        tracks_log[cam] = cam_tracks
                    debug_payload = {
                        "frame": frame_idx,
                        "tracks": tracks_log,
                        "association": assoc_debug,
                    }
                    debug_fp.write(json.dumps(debug_payload) + "\n")

                if write_video:
                    rendered: list[np.ndarray] = []
                    for cam in cam_names:
                        rendered.append(_draw_tracks(frames[cam], per_cam_tracks[cam], c.targets, color_seed))
                    h = min(img.shape[0] for img in rendered)
                    resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for img in rendered]
                    combined = cv2.hconcat(resized)
                    if writer is None:
                        fps_cfg = c.video_fps
                        if fps_cfg > 0.0:
                            fps = fps_cfg
                        elif sync_fps > 0.0:
                            fps = sync_fps
                        else:
                            fps_candidates = [info.fps for info in infos.values() if info.fps > 0.0]
                            fps = max(fps_candidates) if fps_candidates else 30.0
                        writer = cv2.VideoWriter(
                            str(video_out_path),
                            cv2.VideoWriter_fourcc(*"XVID"),
                            float(fps),
                            (combined.shape[1], combined.shape[0]),
                        )
                    writer.write(combined)

            out_path = output_root / f"{group_name}.json"
            out_path.write_text(json.dumps(out_json, indent=2, default=str))
            print(f"✅ Group {group_name}: wrote {out_path}" + (f" and {video_out_path}" if write_video else ""))
        finally:
            for cap in caps.values():
                cap.release()
            if writer is not None:
                writer.release()
            if debug_fp is not None:
                debug_fp.close()
