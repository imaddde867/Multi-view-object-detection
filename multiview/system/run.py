from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from multiview.labeling.proposals import parse_kv_map
from multiview.tracking.embeddings import ColorHistEmbedder, TorchResNet18Embedder
from multiview.tracking.global_association import GlobalIDAssigner, GlobalTrackView
from multiview.tracking.simple_tracker import SimpleTracker
from multiview.utils.boxes import Det
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


class YoloDetector:
    def __init__(self, weights: str, *, device: str = "", half: bool = False):
        weights_path = Path(weights)
        if weights_path.suffix:
            if weights_path.is_absolute() or weights_path.parent != Path("."):
                if not weights_path.exists():
                    raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
            elif not weights_path.exists():
                print(
                    "⚠️ YOLO weights not found locally. Ultralytics will try to download them. "
                    "If you're offline, download weights and pass a local path."
                )
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: ultralytics. Install with `pip install -r requirements.txt`.") from e

        self.device = str(device or "").strip()
        self.half = bool(half)
        self.model = YOLO(weights)
        if self.device:
            try:
                self.model.to(self.device)
            except Exception:
                pass

        self.names: dict[int, str] = {}
        names_obj: Any = getattr(self.model, "names", None)
        if isinstance(names_obj, dict):
            self.names = {int(k): str(v) for k, v in names_obj.items()}
        elif isinstance(names_obj, (list, tuple)):
            self.names = {i: str(n) for i, n in enumerate(names_obj)}

    def predict_batch(
        self, frames_bgr: list[np.ndarray], *, conf: float, iou: float, imgsz: int
    ) -> list[list[tuple[tuple[float, float, float, float], float, int]]]:
        results = self.model.predict(
            frames_bgr,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            half=bool(self.half),
            verbose=False,
        )
        out: list[list[tuple[tuple[float, float, float, float], float, int]]] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append([])
                continue
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = boxes.conf.cpu().numpy().astype(np.float32)
            cls_ids = boxes.cls.cpu().numpy().astype(np.int32)
            dets = []
            for b, s, c in zip(xyxy, scores, cls_ids):
                x0, y0, x1, y1 = [float(v) for v in b.tolist()]
                dets.append(((x0, y0, x1, y1), float(s), int(c)))
            out.append(dets)
        return out


def _parse_targets(targets: Any) -> list[str]:
    if isinstance(targets, list):
        return [str(t) for t in targets]
    return [t.strip() for t in str(targets).split(",") if t.strip()]


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


def run_multiview(cfg: dict[str, Any]) -> None:
    base = _default_cfg()
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v

    cameras_cfg = base.get("cameras", {})
    groups_cfg = base.get("groups", {})
    if not cameras_cfg or not groups_cfg:
        raise ValueError("Config must define `cameras` and `groups`.")

    detector_cfg = base["detector"]
    tracker_cfg = base["tracker"]
    run_cfg = base["run"]
    out_cfg = base["output"]
    runtime_cfg = base["runtime"]

    targets = _parse_targets(detector_cfg.get("targets", "person,car"))
    target_to_id = {n: i for i, n in enumerate(targets)}
    source_map = parse_kv_map(detector_cfg.get("source_map", ""))

    model_path = str(detector_cfg.get("model", "")).strip()
    if not model_path:
        raise ValueError("Missing detector.model in config (weights path).")

    detector = YoloDetector(
        model_path,
        device=str(runtime_cfg.get("device", "")),
        half=bool(runtime_cfg.get("half", False)),
    )

    embedder_cfg = tracker_cfg.get("embedder", {}) or {}
    if isinstance(embedder_cfg, str):
        embedder_cfg = {"type": embedder_cfg}
    embedder_type = str(embedder_cfg.get("type", "colorhist"))
    if embedder_type == "torch_resnet18":
        embedder_device = str(embedder_cfg.get("device", runtime_cfg.get("device", "cpu")) or "cpu")
        if embedder_device.isdigit():
            embedder_device = f"cuda:{embedder_device}"
        embedder = TorchResNet18Embedder(device=embedder_device)
    else:
        bins = int(embedder_cfg.get("bins", 16))
        embedder = ColorHistEmbedder(bins=bins)

    selected_groups: list[str] = [str(g) for g in run_cfg.get("groups", []) if str(g).strip()]
    if not selected_groups:
        selected_groups = [str(g) for g in groups_cfg.keys()]

    output_root = Path(str(out_cfg.get("dir", "results/system")))
    output_root.mkdir(parents=True, exist_ok=True)

    frame_stride = int(run_cfg.get("frame_stride", 1) or 1)
    max_frames = run_cfg.get("max_frames", None)
    max_frames_i = None if max_frames in (None, "", 0) else int(max_frames)
    debug_cfg = base.get("debug", {}) or {}
    debug_assoc = bool(debug_cfg.get("global_assoc", False))
    debug_log_path = str(debug_cfg.get("log_path", "")).strip()

    cam_homographies: dict[str, np.ndarray] = {}
    for cam, cam_entry in cameras_cfg.items():
        if not isinstance(cam_entry, dict):
            continue
        homography = cam_entry.get("homography")
        if homography is None:
            image_points = cam_entry.get("image_points")
            world_points = cam_entry.get("world_points")
            if image_points is None or world_points is None:
                continue
            img = np.asarray(image_points, dtype=np.float32)
            world = np.asarray(world_points, dtype=np.float32)
            if img.shape != world.shape or img.ndim != 2 or img.shape[0] < 4 or img.shape[1] != 2:
                raise ValueError(
                    f"Camera {cam} image_points/world_points must be Nx2 with N>=4 and matching shapes."
                )
            mat, _ = cv2.findHomography(img, world, method=0)
            if mat is None:
                raise ValueError(f"Camera {cam} homography could not be estimated from points.")
        else:
            mat = np.asarray(homography, dtype=np.float32)
            if mat.shape != (3, 3):
                raise ValueError(f"Camera {cam} homography must be a 3x3 matrix.")
        cam_homographies[str(cam)] = mat.astype(np.float32)

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

            trackers = {
                cam: SimpleTracker(
                    embedder,
                    max_age=int(tracker_cfg.get("max_age", 30)),
                    match_threshold=float(tracker_cfg.get("match_threshold", 0.5)),
                )
                for cam in cam_names
            }
            global_max_age = int(tracker_cfg.get("global_max_age", tracker_cfg.get("max_age", 30)))
            gid = GlobalIDAssigner(
                match_threshold=float(tracker_cfg.get("global_match_threshold", 0.75)),
                max_age=global_max_age,
                spatial_weight=float(tracker_cfg.get("spatial_weight", 0.0)),
                spatial_sigma=float(tracker_cfg.get("spatial_sigma", 1.0)),
                spatial_max_dist=float(tracker_cfg.get("spatial_max_dist", 0.0)),
                embedding_alpha=float(tracker_cfg.get("embedding_alpha", 1.0)),
                world_alpha=float(tracker_cfg.get("world_alpha", 1.0)),
            )
            anchor_cam = cam_names[0]

            out_json: dict[str, Any] = {
                "metadata": {
                    "group": group_name,
                    "cameras": {c: _camera_info(infos[c]) for c in cam_names},
                    "classes": targets,
                    "model": model_path,
                },
                "frames": [],
            }

            write_video = bool(out_cfg.get("write_video", False))
            video_out_path = output_root / f"{group_name}.avi"
            color_seed = 1337
            if debug_assoc:
                debug_path = Path(debug_log_path) if debug_log_path else output_root / f"{group_name}_global_assoc.jsonl"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_fp = debug_path.open("w")

            processed = 0
            raw_idx = -1
            while True:
                raw_idx += 1
                frames: dict[str, np.ndarray] = {}
                ok = True
                for cam in cam_names:
                    ret, frame = caps[cam].read()
                    if not ret:
                        ok = False
                        break
                    frames[cam] = frame
                if not ok:
                    break

                if raw_idx % frame_stride != 0:
                    continue

                processed += 1
                if max_frames_i is not None and processed > max_frames_i:
                    break

                batch = [frames[c] for c in cam_names]
                dets_batch = detector.predict_batch(
                    batch,
                    conf=float(detector_cfg.get("conf", 0.35)),
                    iou=float(detector_cfg.get("iou", 0.5)),
                    imgsz=int(detector_cfg.get("imgsz", 960)),
                )

                per_cam_tracks: dict[str, list[dict[str, Any]]] = {}
                views_by_cam: dict[str, list[GlobalTrackView]] = {}
                views_by_key: dict[tuple[str, int], GlobalTrackView] = {}

                for cam, frame, dets_raw in zip(cam_names, batch, dets_batch):
                    mapped_dets: list[Det] = []
                    for xyxy, score, cls_id in dets_raw:
                        src_name = detector.names.get(int(cls_id), str(int(cls_id)))
                        tgt_name = source_map.get(src_name, src_name)
                        if tgt_name not in target_to_id:
                            continue
                        mapped_dets.append(Det(xyxy=xyxy, cls_id=target_to_id[tgt_name], score=float(score)))

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
                                "cls_name": targets[int(t.cls_id)],
                                "bbox_xyxy": [float(v) for v in t.xyxy],
                                "score": float(t.score),
                            }
                        )
                        views.append(view)
                        views_by_key[(cam, local)] = view
                    per_cam_tracks[cam] = out_tracks
                    views_by_cam[cam] = views

                frame_idx = processed - 1
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
                        rendered.append(_draw_tracks(frames[cam], per_cam_tracks[cam], targets, color_seed))
                    h = min(img.shape[0] for img in rendered)
                    resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for img in rendered]
                    combined = cv2.hconcat(resized)
                    if writer is None:
                        fps_cfg = float(out_cfg.get("video_fps") or 0.0)
                        if fps_cfg > 0.0:
                            fps = fps_cfg
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
