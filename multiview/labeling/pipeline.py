from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from multiview.labeling.proposals import YoloProposer, parse_kv_map
from multiview.labeling.sam3_refiner import Sam3BoxRefiner, Sam3Config
from multiview.utils.boxes import Det, area_xyxy, nms_xyxy, to_yolo_line
from multiview.utils.hash_split import stable_split
from multiview.utils.video import open_video


@dataclass(frozen=True)
class LabelConfig:
    videos: list[str]
    groups: dict[str, dict[str, str]]
    out: str
    targets: list[str]
    proposal_model: str
    proposal_imgsz: int
    source_map: str
    conf: float
    iou: float
    frame_stride: int
    max_frames_per_video: int
    train_ratio: float
    val_ratio: float
    seed: int
    min_box_area: int
    keep_empty: bool
    save_viz: bool

    sam3_checkpoint: str
    sam3_load_from_hf: bool
    sam3_device: str
    sam3_confidence: float
    mask_close: int
    sam3_nms_iou: float

    runtime_device: str
    runtime_half: bool


def _default_cfg() -> dict[str, Any]:
    return {
        "videos": [],
        "groups": {},
        "out": "data/processed/sam3_autolabel_v2",
        "targets": "person,car,bus",
        "proposal_model": "checkpoints/yolo/yolo11m.pt",
        "proposal_imgsz": 1536,
        "source_map": "truck=car,motorcycle=car,bicycle=car",
        "conf": 0.15,
        "iou": 0.7,
        "frame_stride": 5,
        "max_frames_per_video": 4000,
        "train_ratio": 0.9,
        "val_ratio": 0.1,
        "seed": 0,
        "min_box_area": 400,
        "keep_empty": False,
        "save_viz": False,
        "sam3": {
            "checkpoint": "checkpoints/sam3/sam3.pt",
            "load_from_hf": False,
            "device": "cuda",
            "confidence": 0.5,
            "mask_close": 7,
            "nms_iou": 0.5,
        },
        "runtime": {"device": "", "half": False},
    }


def _normalize_groups(raw: Any) -> dict[str, dict[str, str]]:
    """
    Normalize group configuration into:
      {group_name: {camera_name: video_path}}

    Supported YAML shapes:
      groups:
        g12: [path/to/cam1.mp4, path/to/cam2.mp4]
        g34:
          cam3: path/to/cam3.mp4
          cam4: path/to/cam4.mp4
    """
    if raw in (None, "", [], {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("`groups` must be a mapping like {group: [videos] | {cam: video}}")

    out: dict[str, dict[str, str]] = {}
    for group_name_any, group_val in raw.items():
        group_name = str(group_name_any).strip()
        if not group_name:
            raise ValueError("Group name cannot be empty.")

        cams: dict[str, str] = {}
        if isinstance(group_val, str):
            p = str(group_val)
            cam = Path(p).stem
            cams[cam] = p
        elif isinstance(group_val, (list, tuple)):
            for item in group_val:
                if item is None:
                    continue
                p = str(item)
                cam = Path(p).stem
                if cam in cams:
                    raise ValueError(f"Duplicate camera name '{cam}' in group '{group_name}'.")
                cams[cam] = p
        elif isinstance(group_val, dict):
            for cam_name_any, video_path_any in group_val.items():
                cam = str(cam_name_any).strip()
                if not cam:
                    raise ValueError(f"Empty camera name in group '{group_name}'.")
                p = str(video_path_any)
                if cam in cams:
                    raise ValueError(f"Duplicate camera name '{cam}' in group '{group_name}'.")
                cams[cam] = p
        else:
            raise ValueError(f"Invalid group entry for '{group_name}': expected list/dict, got {type(group_val).__name__}")

        if len(cams) < 1:
            raise ValueError(f"Group '{group_name}' must contain at least one video.")
        out[group_name] = cams

    return out


def _as_label_config(cfg: dict[str, Any]) -> LabelConfig:
    base = _default_cfg()
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v

    raw_targets = base["targets"]
    if isinstance(raw_targets, list):
        targets = [str(t).strip() for t in raw_targets if str(t).strip()]
    else:
        targets = [t.strip() for t in str(raw_targets).split(",") if t.strip()]
    videos = base["videos"] or []
    if isinstance(videos, str):
        videos = [videos]

    groups = _normalize_groups(base.get("groups"))

    sam3 = base["sam3"]
    runtime = base["runtime"]

    proposal_imgsz_raw = base.get("proposal_imgsz", 960)
    proposal_imgsz = int(proposal_imgsz_raw) if proposal_imgsz_raw not in (None, "") else 0
    if proposal_imgsz < 0:
        proposal_imgsz = 0

    frame_stride = int(base.get("frame_stride", 1) or 1)
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")

    max_frames_raw = base.get("max_frames_per_video", 0)
    if max_frames_raw in (None, "", 0):
        max_frames_per_video = 0
    else:
        max_frames_per_video = int(max_frames_raw)
        if max_frames_per_video < 0:
            max_frames_per_video = 0

    train_ratio = float(base.get("train_ratio", 0.0))
    val_ratio = float(base.get("val_ratio", 0.0))
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0 + 1e-9:
        raise ValueError("Expected train_ratio>=0, val_ratio>=0 and train_ratio+val_ratio<=1.")

    min_box_area = int(base.get("min_box_area", 0) or 0)
    if min_box_area < 0:
        raise ValueError("min_box_area must be >= 0.")

    return LabelConfig(
        videos=[str(v) for v in videos],
        groups=groups,
        out=str(base["out"]),
        targets=targets,
        proposal_model=str(base["proposal_model"]),
        proposal_imgsz=proposal_imgsz,
        source_map=str(base["source_map"]),
        conf=float(base["conf"]),
        iou=float(base["iou"]),
        frame_stride=frame_stride,
        max_frames_per_video=max_frames_per_video,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=int(base["seed"]),
        min_box_area=min_box_area,
        keep_empty=bool(base["keep_empty"]),
        save_viz=bool(base["save_viz"]),
        sam3_checkpoint=str(sam3.get("checkpoint", "")),
        sam3_load_from_hf=bool(sam3.get("load_from_hf", False)),
        sam3_device=str(sam3.get("device", "cuda")),
        sam3_confidence=float(sam3.get("confidence", 0.5)),
        mask_close=int(sam3.get("mask_close", 0)),
        sam3_nms_iou=float(sam3.get("nms_iou", 0.5)),
        runtime_device=str(runtime.get("device", "")),
        runtime_half=bool(runtime.get("half", False)),
    )


def _ensure_dataset_dirs(out_root: Path, save_viz: bool) -> None:
    for subset in ("train", "val", "test"):
        (out_root / subset / "images").mkdir(parents=True, exist_ok=True)
        (out_root / subset / "labels").mkdir(parents=True, exist_ok=True)
        if save_viz:
            (out_root / "viz" / subset).mkdir(parents=True, exist_ok=True)


def _draw_viz_overlay(
    frame_bgr: np.ndarray,
    *,
    proposal_dets: list[Det],
    refined_dets: list[Det],
    names: list[str],
) -> np.ndarray:
    out = frame_bgr.copy()
    # Proposals in red, refined in green.
    for det in proposal_dets:
        x0, y0, x1, y1 = [int(v) for v in det.xyxy]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        label = f"P:{names[det.cls_id]}"
        cv2.putText(out, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for det in refined_dets:
        x0, y0, x1, y1 = [int(v) for v in det.xyxy]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"R:{names[det.cls_id]}"
        cv2.putText(out, label, (x0, min(frame_bgr.shape[0] - 1, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def label_videos(cfg: dict[str, Any]) -> None:
    c = _as_label_config(cfg)
    if not c.videos and not c.groups:
        raise ValueError("No videos provided. Set `videos`/`groups` in config or pass `--videos`.")
    if not c.targets:
        raise ValueError("Targets cannot be empty.")
    if len(set(c.targets)) != len(c.targets):
        raise ValueError("Targets must be unique.")
    if "cuda" in c.sam3_device.lower():
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required for SAM3 labeling, but no GPU is visible.\n"
                    "You're likely on a login/CPU node. On Puhti, run via:\n"
                    "  `sbatch slurm/label.sbatch`\n"
                    "or an interactive GPU shell:\n"
                    "  `srun -A project_2015432 -p gpu --gres=gpu:v100:1 --cpus-per-task=16 --mem=128G --time=00:20:00 --pty bash -l`"
                )
        except ModuleNotFoundError:
            raise RuntimeError("Missing dependency: torch (required for SAM3 labeling).") from None

    out_root = Path(c.out)
    _ensure_dataset_dirs(out_root, save_viz=c.save_viz)

    target_to_id = {name: i for i, name in enumerate(c.targets)}
    source_map = parse_kv_map(c.source_map)

    proposer = YoloProposer(c.proposal_model, device=c.runtime_device, half=c.runtime_half)
    sam3 = Sam3BoxRefiner(
        Sam3Config(
            checkpoint=c.sam3_checkpoint,
            load_from_hf=c.sam3_load_from_hf,
            device=c.sam3_device,
            confidence=c.sam3_confidence,
            mask_close=c.mask_close,
        )
    )

    meta_path = out_root / "meta.json"
    meta = {
        "targets": c.targets,
        "proposal_model": c.proposal_model,
        "proposal_imgsz": c.proposal_imgsz,
        "source_map": source_map,
        "conf": c.conf,
        "iou": c.iou,
        "frame_stride": c.frame_stride,
        "seed": c.seed,
        "sam3": {
            "checkpoint": c.sam3_checkpoint,
            "load_from_hf": c.sam3_load_from_hf,
            "device": c.sam3_device,
            "confidence": c.sam3_confidence,
            "mask_close": c.mask_close,
            "nms_iou": c.sam3_nms_iou,
        },
        "runtime": {"device": c.runtime_device, "half": c.runtime_half},
    }
    if c.groups:
        meta["groups"] = c.groups
    else:
        meta["videos"] = c.videos
    meta_path.write_text(json.dumps(meta, indent=2))

    total_images = 0
    total_labels = 0

    def _iter_group_frames(caps: dict[str, cv2.VideoCapture], *, stride: int, max_frames: int) -> Any:
        stride = max(1, int(stride))
        kept = 0
        raw_idx = -1
        while True:
            raw_idx += 1
            frames: dict[str, np.ndarray] = {}
            ok = True
            for cam, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    ok = False
                    break
                frames[cam] = frame
            if not ok:
                break
            if raw_idx % stride != 0:
                continue
            yield raw_idx, frames
            kept += 1
            if max_frames and kept >= max_frames:
                break

    groups: dict[str, dict[str, str]]
    if c.groups:
        groups = c.groups
    else:
        groups = {Path(v).stem: {Path(v).stem: v} for v in c.videos}

    for group_name, cam_to_video in groups.items():
        caps: dict[str, cv2.VideoCapture] = {}
        pbar: tqdm | None = None
        try:
            for cam, video in cam_to_video.items():
                cap, _info = open_video(Path(video))
                caps[cam] = cap

            warned_sam3_fallback = False
            pbar = tqdm(
                _iter_group_frames(caps, stride=c.frame_stride, max_frames=c.max_frames_per_video),
                desc=f"Labeling {group_name}",
                unit="frame",
            )
            for frame_idx, frames in pbar:
                split_key = f"{group_name}_f{frame_idx:06d}"
                subset = stable_split(split_key, c.seed, c.train_ratio, c.val_ratio)

                per_cam: dict[str, dict[str, Any]] = {}
                any_labels = False

                for cam, frame in frames.items():
                    h, w = frame.shape[:2]

                    proposals = proposer.propose(
                        frame,
                        conf=c.conf,
                        iou=c.iou,
                        target_to_id=target_to_id,
                        source_map=source_map,
                        min_box_area=c.min_box_area,
                        imgsz=c.proposal_imgsz if c.proposal_imgsz > 0 else None,
                    )

                    dets: list[Det] = []
                    proposal_boxes = [p.det.xyxy for p in proposals]
                    prompts = [p.tgt_name for p in proposals]
                    refined = proposal_boxes
                    if proposal_boxes:
                        try:
                            refined = sam3.refine_xyxy_with_prompts(frame, proposal_boxes, prompts)
                        except Exception as e:
                            if not warned_sam3_fallback:
                                print(f"⚠️ SAM3 refinement failed for group {group_name} (falling back to proposal boxes): {e}")
                                warned_sam3_fallback = True
                            refined = proposal_boxes

                    for p, xyxy in zip(proposals, refined):
                        if area_xyxy(xyxy) < float(c.min_box_area):
                            continue
                        dets.append(Det(xyxy=xyxy, cls_id=p.det.cls_id, score=p.det.score))

                    if c.sam3_nms_iou > 0 and dets:
                        by_cls: dict[int, list[Det]] = {}
                        for d in dets:
                            by_cls.setdefault(d.cls_id, []).append(d)
                        dets = []
                        for cls_id, cls_dets in by_cls.items():
                            dets.extend(nms_xyxy(cls_dets, iou_thr=float(c.sam3_nms_iou)))

                    lines: list[str] = []
                    for d in dets:
                        line = to_yolo_line(d.cls_id, d.xyxy, width=w, height=h)
                        if line is not None:
                            lines.append(line)

                    if lines:
                        any_labels = True

                    per_cam[cam] = {"frame": frame, "proposals": proposals, "dets": dets, "lines": lines}

                if not any_labels and not c.keep_empty:
                    continue

                for cam, item in per_cam.items():
                    name = f"{group_name}_{cam}_f{frame_idx:06d}"
                    frame = item["frame"]
                    lines = item["lines"]

                    img_out = out_root / subset / "images" / f"{name}.jpg"
                    lbl_out = out_root / subset / "labels" / f"{name}.txt"
                    cv2.imwrite(str(img_out), frame)
                    lbl_out.write_text("\n".join(lines) + ("\n" if lines else ""))
                    total_images += 1
                    total_labels += len(lines)

                    if c.save_viz:
                        proposal_dets = [p.det for p in item["proposals"]]
                        viz = _draw_viz_overlay(frame, proposal_dets=proposal_dets, refined_dets=item["dets"], names=c.targets)
                        viz_out = out_root / "viz" / subset / f"{name}.jpg"
                        cv2.imwrite(str(viz_out), viz)

                pbar.set_postfix(images=total_images, labels=total_labels)
        finally:
            if pbar is not None:
                pbar.close()
            for cap in caps.values():
                cap.release()

    dataset_yaml = out_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                # Keep dataset portable across machines / scratch paths.
                # Ultralytics resolves relative `path` relative to this YAML file.
                "path: .",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "",
                f"nc: {len(c.targets)}",
                f"names: {c.targets}",
                "",
            ]
        )
    )

    stats = {"images": total_images, "labels": total_labels, "config": asdict(c)}
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"✅ Labeled dataset written to: {dataset_yaml}")
