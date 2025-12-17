from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from mvot.labeling.proposals import YoloProposer, parse_kv_map
from mvot.labeling.sam3_refiner import Sam3BoxRefiner, Sam3Config
from mvot.utils.boxes import Det, area_xyxy, nms_xyxy, to_yolo_line
from mvot.utils.hash_split import stable_split
from mvot.utils.video import iter_frames, open_video


@dataclass(frozen=True)
class LabelConfig:
    videos: list[str]
    out: str
    targets: list[str]
    proposal_model: str
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
        "out": "data/processed/sam3_autolabel",
        "targets": "person,car,bus",
        "proposal_model": "yolov8n.pt",
        "source_map": "truck=car,motorcycle=car,bicycle=car",
        "conf": 0.35,
        "iou": 0.5,
        "frame_stride": 5,
        "max_frames_per_video": 4000,
        "train_ratio": 0.9,
        "val_ratio": 0.1,
        "seed": 0,
        "min_box_area": 400,
        "keep_empty": False,
        "save_viz": False,
        "sam3": {
            "checkpoint": "",
            "load_from_hf": False,
            "device": "cuda",
            "confidence": 0.5,
            "mask_close": 0,
            "nms_iou": 0.5,
        },
        "runtime": {"device": "", "half": False},
    }


def _as_label_config(cfg: dict[str, Any]) -> LabelConfig:
    base = _default_cfg()
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v

    targets = [t.strip() for t in str(base["targets"]).split(",") if t.strip()]
    videos = base["videos"] or []
    if isinstance(videos, str):
        videos = [videos]

    sam3 = base["sam3"]
    runtime = base["runtime"]
    return LabelConfig(
        videos=[str(v) for v in videos],
        out=str(base["out"]),
        targets=targets,
        proposal_model=str(base["proposal_model"]),
        source_map=str(base["source_map"]),
        conf=float(base["conf"]),
        iou=float(base["iou"]),
        frame_stride=int(base["frame_stride"]),
        max_frames_per_video=int(base["max_frames_per_video"]),
        train_ratio=float(base["train_ratio"]),
        val_ratio=float(base["val_ratio"]),
        seed=int(base["seed"]),
        min_box_area=int(base["min_box_area"]),
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


def _draw_viz(frame_bgr: np.ndarray, dets: list[Det], names: list[str]) -> np.ndarray:
    out = frame_bgr.copy()
    for det in dets:
        x0, y0, x1, y1 = [int(v) for v in det.xyxy]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{names[det.cls_id]} {det.score:.2f}"
        cv2.putText(out, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def label_videos(cfg: dict[str, Any]) -> None:
    c = _as_label_config(cfg)
    if not c.videos:
        raise ValueError("No videos provided. Set `videos` in config or pass `--videos`.")
    if not c.targets:
        raise ValueError("Targets cannot be empty.")

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
    meta_path.write_text(json.dumps(meta, indent=2))

    total_images = 0
    total_labels = 0

    for video in c.videos:
        video_path = Path(video)
        cap, info = open_video(video_path)
        stem = video_path.stem
        warned_sam3_fallback = False
        pbar = tqdm(iter_frames(cap, stride=c.frame_stride, max_frames=c.max_frames_per_video), desc=f"Labeling {stem}", unit="frame")
        for frame_idx, frame in pbar:
            name = f"{stem}_f{frame_idx:06d}"
            subset = stable_split(name, c.seed, c.train_ratio, c.val_ratio)
            h, w = frame.shape[:2]

            proposals = proposer.propose(
                frame,
                conf=c.conf,
                iou=c.iou,
                target_to_id=target_to_id,
                source_map=source_map,
                min_box_area=c.min_box_area,
            )
            if not proposals:
                if not c.keep_empty:
                    continue
                img_out = out_root / subset / "images" / f"{name}.jpg"
                lbl_out = out_root / subset / "labels" / f"{name}.txt"
                cv2.imwrite(str(img_out), frame)
                lbl_out.write_text("")
                total_images += 1
                continue

            boxes = [p.det.xyxy for p in proposals]
            prompts = [p.tgt_name for p in proposals]
            try:
                refined = sam3.refine_xyxy_with_prompts(frame, boxes, prompts)
            except Exception as e:
                if not warned_sam3_fallback:
                    print(f"⚠️ SAM3 refinement failed for {stem} (falling back to proposal boxes): {e}")
                    warned_sam3_fallback = True
                refined = boxes

            dets: list[Det] = []
            for p, xyxy in zip(proposals, refined):
                if area_xyxy(xyxy) < float(c.min_box_area):
                    continue
                dets.append(Det(xyxy=xyxy, cls_id=p.det.cls_id, score=p.det.score))

            if c.sam3_nms_iou > 0:
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
            if not lines and not c.keep_empty:
                continue

            img_out = out_root / subset / "images" / f"{name}.jpg"
            lbl_out = out_root / subset / "labels" / f"{name}.txt"
            cv2.imwrite(str(img_out), frame)
            lbl_out.write_text("\n".join(lines) + ("\n" if lines else ""))
            total_images += 1
            total_labels += len(lines)

            if c.save_viz:
                viz = _draw_viz(frame, dets, c.targets)
                viz_out = out_root / "viz" / subset / f"{name}.jpg"
                cv2.imwrite(str(viz_out), viz)

            pbar.set_postfix(images=total_images, labels=total_labels)

        cap.release()

    dataset_yaml = out_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve()}",
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
