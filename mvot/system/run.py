from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mvot.labeling.proposals import parse_kv_map
from mvot.tracking.embeddings import ColorHistEmbedder, TorchResNet18Embedder
from mvot.tracking.global_association import GlobalIDAssigner, GlobalTrackView
from mvot.tracking.simple_tracker import SimpleTracker
from mvot.utils.boxes import Det
from mvot.utils.video import VideoInfo, open_video


@dataclass(frozen=True)
class CameraCfg:
    name: str
    source: str


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
            "embedder": {"type": "colorhist"},
        },
        "run": {"frame_stride": 1, "max_frames": None, "groups": []},
        "output": {"dir": "results/system", "write_video": False, "video_fps": 0.0},
        "runtime": {"device": "", "half": False},
    }


class YoloDetector:
    def __init__(self, weights: str, *, device: str = "", half: bool = False):
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
        cls_id = int(tr["cls_id"])
        x0, y0, x1, y1 = [int(v) for v in tr["bbox_xyxy"]]
        rng = np.random.default_rng(color_seed + gid * 97)
        color = tuple(int(x) for x in rng.integers(20, 235, size=3))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"ID:{gid} {class_names[cls_id]}"
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

    embedder_type = str(tracker_cfg.get("embedder", {}).get("type", "colorhist"))
    if embedder_type == "torch_resnet18":
        embedder = TorchResNet18Embedder(device=str(runtime_cfg.get("device", "cpu")) or "cpu")
    else:
        embedder = ColorHistEmbedder()

    selected_groups: list[str] = [str(g) for g in run_cfg.get("groups", []) if str(g).strip()]
    if not selected_groups:
        selected_groups = [str(g) for g in groups_cfg.keys()]

    output_root = Path(str(out_cfg.get("dir", "results/system")))
    output_root.mkdir(parents=True, exist_ok=True)

    frame_stride = int(run_cfg.get("frame_stride", 1) or 1)
    max_frames = run_cfg.get("max_frames", None)
    max_frames_i = None if max_frames in (None, "", 0) else int(max_frames)

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
        gid = GlobalIDAssigner(match_threshold=float(tracker_cfg.get("global_match_threshold", 0.75)))
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
        writer: cv2.VideoWriter | None = None
        video_out_path = output_root / f"{group_name}.avi"
        color_seed = 1337

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
            views_anchor: list[GlobalTrackView] = []
            views_others: dict[str, list[GlobalTrackView]] = {}

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
                    global_id = gid.ensure(cam, local)
                    out_tracks.append(
                        {
                            "global_id": global_id,
                            "local_id": local,
                            "cls_id": int(t.cls_id),
                            "cls_name": targets[int(t.cls_id)],
                            "bbox_xyxy": [float(v) for v in t.xyxy],
                            "score": float(t.score),
                        }
                    )
                    views.append(
                        GlobalTrackView(
                            cam=cam,
                            local_id=local,
                            cls_id=int(t.cls_id),
                            xyxy=t.xyxy,
                            embedding=t.embedding,
                        )
                    )
                per_cam_tracks[cam] = out_tracks

                if cam == anchor_cam:
                    views_anchor = views
                else:
                    views_others[cam] = views

            for other_cam, other_views in views_others.items():
                gid.associate_anchor(anchor_cam, views_anchor, other_cam, other_views)

            for cam in cam_names:
                for tr in per_cam_tracks[cam]:
                    tr["global_id"] = gid.ensure(cam, int(tr["local_id"]))

            out_json["frames"].append({"frame": processed - 1, "cameras": per_cam_tracks})

            if write_video:
                rendered: list[np.ndarray] = []
                for cam in cam_names:
                    rendered.append(_draw_tracks(frames[cam], per_cam_tracks[cam], targets, color_seed))
                h = min(img.shape[0] for img in rendered)
                resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for img in rendered]
                combined = cv2.hconcat(resized)
                if writer is None:
                    fps = float(out_cfg.get("video_fps") or 0.0) or infos[anchor_cam].fps or 30.0
                    writer = cv2.VideoWriter(
                        str(video_out_path),
                        cv2.VideoWriter_fourcc(*"XVID"),
                        float(fps),
                        (combined.shape[1], combined.shape[0]),
                    )
                writer.write(combined)

        for cap in caps.values():
            cap.release()
        if writer is not None:
            writer.release()

        out_path = output_root / f"{group_name}.json"
        out_path.write_text(json.dumps(out_json, indent=2, default=str))
        print(f"✅ Group {group_name}: wrote {out_path}" + (f" and {video_out_path}" if write_video else ""))
