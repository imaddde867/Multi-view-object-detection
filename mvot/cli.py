from __future__ import annotations

import argparse
from pathlib import Path

from mvot.dataset.verify import verify_dataset
from mvot.labeling.pipeline import label_videos
from mvot.system.run import run_multiview
from mvot.training.train import train_yolo
from mvot.utils.yaml import load_yaml, merge_dicts


def _add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., cuda:0, 0, cpu).")
    # Tri-state: None preserves config, True overrides.
    parser.add_argument("--half", action="store_true", default=None, help="Use half precision (CUDA only).")


def _cmd_label(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", type=str, default="", help="Optional YAML config; CLI args override it.")
    sub.add_argument("--videos", nargs="+", required=False, help="Input video paths.")
    sub.add_argument("--out", type=str, default="", help="Output dataset directory.")
    sub.add_argument("--targets", type=str, default="", help="Comma-separated targets (default: person,car,bus).")
    sub.add_argument("--proposal-model", type=str, default="", help="Ultralytics weights for class proposals.")
    sub.add_argument("--proposal-imgsz", type=int, default=-1, help="Inference size for proposal model.")
    sub.add_argument("--source-map", type=str, default="", help="Optional mapping (e.g. 'truck=car,motorcycle=car').")
    sub.add_argument("--conf", type=float, default=-1.0, help="Proposal confidence threshold.")
    sub.add_argument("--iou", type=float, default=-1.0, help="Proposal IoU threshold.")
    sub.add_argument("--frame-stride", type=int, default=-1, help="Keep 1 frame every N frames.")
    sub.add_argument("--max-frames-per-video", type=int, default=-1, help="Safety cap per video.")
    sub.add_argument("--train-ratio", type=float, default=-1.0, help="Train split ratio.")
    sub.add_argument("--val-ratio", type=float, default=-1.0, help="Val split ratio.")
    sub.add_argument("--seed", type=int, default=-1, help="Split seed.")
    sub.add_argument("--min-box-area", type=int, default=-1, help="Drop boxes smaller than this (pixels^2).")
    # Tri-state: None preserves config, True overrides.
    sub.add_argument("--save-viz", action="store_true", default=None, help="Save visualization images.")
    sub.add_argument("--keep-empty", action="store_true", default=None, help="Keep frames even if no labels found.")
    sub.add_argument("--sam3-checkpoint", type=str, default="", help="Path to SAM3 checkpoint.")
    # Tri-state: None preserves config, True overrides.
    sub.add_argument("--sam3-load-from-hf", action="store_true", default=None, help="Allow SAM3 to download checkpoints.")
    sub.add_argument("--sam3-confidence", type=float, default=-1.0, help="SAM3 confidence threshold.")
    sub.add_argument("--mask-close", type=int, default=-1, help="Morphological close kernel size (odd int).")
    sub.add_argument("--sam3-nms-iou", type=float, default=-1.0, help="NMS IoU for refined boxes.")
    _add_common_io_args(sub)


def _cmd_train(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", type=str, default="", help="Optional YAML config; CLI args override it.")
    sub.add_argument("--data", type=str, default="", help="Path to dataset.yaml.")
    sub.add_argument("--model", type=str, default="", help="Model weights (e.g., yolo11m.pt).")
    sub.add_argument("--epochs", type=int, default=-1, help="Epochs.")
    sub.add_argument("--imgsz", type=int, default=-1, help="Image size.")
    sub.add_argument("--batch", type=int, default=-1, help="Batch size.")
    sub.add_argument("--project", type=str, default="", help="Output project directory (runs).")
    sub.add_argument("--name", type=str, default="", help="Run name.")
    sub.add_argument("--seed", type=int, default=-1, help="Seed.")
    sub.add_argument("--workers", type=int, default=-1, help="Dataloader workers.")
    _add_common_io_args(sub)


def _cmd_run(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", type=str, required=True, help="System YAML config.")
    sub.add_argument("--groups", type=str, default="", help="Comma-separated group names to run (default: all).")
    sub.add_argument("--out", type=str, default="", help="Output directory override.")
    sub.add_argument("--model", type=str, default="", help="Detector weights override.")
    sub.add_argument("--conf", type=float, default=-1.0, help="Detection confidence override.")
    sub.add_argument("--iou", type=float, default=-1.0, help="Detection IoU override.")
    sub.add_argument("--imgsz", type=int, default=-1, help="Inference size override.")
    sub.add_argument("--max-frames", type=int, default=-1, help="Optional max frames per camera.")
    # Tri-state: None preserves config, True overrides.
    sub.add_argument("--write-video", action="store_true", default=None, help="Write rendered video(s).")
    _add_common_io_args(sub)


def _cmd_verify(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--dataset", type=str, required=True, help="Path to dataset root or dataset.yaml.")
    sub.add_argument(
        "--expected-names",
        type=str,
        default="person,car",
        help="Comma-separated expected class names (default: person,car).",
    )
    sub.add_argument("--no-pair-check", action="store_true", help="Skip multi-view pair consistency checks.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mvot", description="SAM3 → YOLO → multi-view detect+track pipeline.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    label_parser = subparsers.add_parser("label", help="Generate YOLO dataset from videos using YOLO proposals + SAM3 refinement.")
    _cmd_label(label_parser)

    train_parser = subparsers.add_parser("train", help="Train a YOLO model on a dataset.yaml.")
    _cmd_train(train_parser)

    run_parser = subparsers.add_parser("run", help="Run multi-view detection + tracking using a system config.")
    _cmd_run(run_parser)

    verify_parser = subparsers.add_parser("verify", help="Verify a YOLO dataset (format + multi-view consistency).")
    _cmd_verify(verify_parser)

    args = parser.parse_args()

    if args.cmd == "label":
        config = {}
        if args.config:
            config = load_yaml(Path(args.config))

        targets = args.targets.strip() if isinstance(args.targets, str) else ""
        out_dir = args.out.strip() if isinstance(args.out, str) else ""
        proposal_model = args.proposal_model.strip() if isinstance(args.proposal_model, str) else ""
        source_map = args.source_map.strip() if isinstance(args.source_map, str) else ""
        device = args.device.strip() if isinstance(args.device, str) else ""

        overrides = {
            "videos": args.videos or None,
            "out": out_dir or None,
            "targets": targets or None,
            "proposal_model": proposal_model or None,
            "proposal_imgsz": args.proposal_imgsz if args.proposal_imgsz >= 1 else None,
            "source_map": source_map or None,
            "conf": args.conf if args.conf >= 0 else None,
            "iou": args.iou if args.iou >= 0 else None,
            "frame_stride": args.frame_stride if args.frame_stride >= 1 else None,
            "max_frames_per_video": args.max_frames_per_video if args.max_frames_per_video >= 1 else None,
            "train_ratio": args.train_ratio if args.train_ratio >= 0 else None,
            "val_ratio": args.val_ratio if args.val_ratio >= 0 else None,
            "seed": args.seed if args.seed >= 0 else None,
            "min_box_area": args.min_box_area if args.min_box_area >= 0 else None,
            "save_viz": args.save_viz,
            "keep_empty": args.keep_empty,
            "sam3": {
                "checkpoint": args.sam3_checkpoint.strip() or None,
                "load_from_hf": args.sam3_load_from_hf,
                "confidence": args.sam3_confidence if args.sam3_confidence >= 0 else None,
                "mask_close": args.mask_close if args.mask_close >= 0 else None,
                "nms_iou": args.sam3_nms_iou if args.sam3_nms_iou >= 0 else None,
            },
            "runtime": {"device": device or None, "half": args.half},
        }
        cfg = merge_dicts(config, overrides)
        label_videos(cfg)
        return

    if args.cmd == "train":
        config = {}
        if args.config:
            config = load_yaml(Path(args.config))

        data_path = args.data.strip() if isinstance(args.data, str) else ""
        model_path = args.model.strip() if isinstance(args.model, str) else ""
        project = args.project.strip() if isinstance(args.project, str) else ""
        name = args.name.strip() if isinstance(args.name, str) else ""
        device = args.device.strip() if isinstance(args.device, str) else ""

        overrides = {
            "data": data_path or None,
            "model": model_path or None,
            "epochs": args.epochs if args.epochs >= 0 else None,
            "imgsz": args.imgsz if args.imgsz >= 0 else None,
            "batch": args.batch if args.batch >= 0 else None,
            "project": project or None,
            "name": name or None,
            "seed": args.seed if args.seed >= 0 else None,
            "workers": args.workers if args.workers >= 0 else None,
            "runtime": {"device": device or None, "half": args.half},
        }
        cfg = merge_dicts(config, overrides)
        train_yolo(cfg)
        return

    if args.cmd == "run":
        cfg = load_yaml(Path(args.config))
        if args.groups:
            requested = [g.strip() for g in args.groups.split(",") if g.strip()]
            run_groups: list[str] = []
            for g in requested:
                if "+" in g:
                    cams = [c.strip() for c in g.split("+") if c.strip()]
                    if len(cams) < 2:
                        raise SystemExit(f"Invalid --groups entry: {g}")
                    cfg.setdefault("groups", {})
                    synthetic = f"group_{'_'.join(cams)}"
                    cfg["groups"][synthetic] = cams
                    run_groups.append(synthetic)
                else:
                    run_groups.append(g)
            cfg["run"] = cfg.get("run", {})
            cfg["run"]["groups"] = run_groups
        if args.out:
            cfg["output"] = cfg.get("output", {})
            cfg["output"]["dir"] = args.out
        if args.model:
            cfg["detector"] = cfg.get("detector", {})
            cfg["detector"]["model"] = args.model
        if args.conf >= 0:
            cfg["detector"] = cfg.get("detector", {})
            cfg["detector"]["conf"] = float(args.conf)
        if args.iou >= 0:
            cfg["detector"] = cfg.get("detector", {})
            cfg["detector"]["iou"] = float(args.iou)
        if args.imgsz >= 0:
            cfg["detector"] = cfg.get("detector", {})
            cfg["detector"]["imgsz"] = int(args.imgsz)
        if args.max_frames >= 0:
            cfg["run"] = cfg.get("run", {})
            cfg["run"]["max_frames"] = int(args.max_frames)
        if args.write_video:
            cfg["output"] = cfg.get("output", {})
            cfg["output"]["write_video"] = True
        cfg["runtime"] = cfg.get("runtime", {})
        if args.device:
            cfg["runtime"]["device"] = args.device
        if args.half:
            cfg["runtime"]["half"] = True
        run_multiview(cfg)
        return

    if args.cmd == "verify":
        raw_expected = str(args.expected_names).strip()
        expected = None if not raw_expected else [t.strip() for t in raw_expected.split(",") if t.strip()]
        report = verify_dataset(args.dataset, expected_names=expected, check_pairs=not bool(args.no_pair_check))
        for w in report.warnings:
            print(f"⚠️ {w}")
        if report.errors:
            for e in report.errors:
                print(f"❌ {e}")
            raise SystemExit(f"Dataset verification failed ({len(report.errors)} error(s)).")

        print(
            "✅ Dataset OK\n"
            f"  root: {report.root}\n"
            f"  names: {report.names}\n"
            f"  images: {report.images}\n"
            f"  empty labels: {report.empty_labels}\n"
            f"  boxes: {report.boxes_total}\n"
            f"  boxes/class: {report.boxes_per_class}"
        )
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
