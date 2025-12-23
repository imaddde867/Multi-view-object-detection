from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multiview.utils.yaml import load_yaml

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class VerifyReport:
    dataset_yaml: Path
    root: Path
    names: list[str]
    images: int
    empty_labels: int
    boxes_total: int
    boxes_per_class: dict[str, int]
    warnings: list[str]
    errors: list[str]


def _as_names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        items: list[tuple[int, str]] = []
        for k, v in raw.items():
            try:
                idx = int(k)
            except Exception:
                continue
            items.append((idx, str(v)))
        return [name for _idx, name in sorted(items, key=lambda x: x[0])]
    if raw is None:
        return []
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _resolve_dataset_yaml(dataset: Path) -> Path:
    if dataset.is_dir():
        return dataset / "dataset.yaml"
    return dataset


def _resolve_root(dataset_yaml: Path, cfg: dict[str, Any]) -> Path:
    raw = cfg.get("path", ".")
    if raw in (None, ""):
        raw = "."
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return (dataset_yaml.parent / p).resolve()


def _iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    out: list[Path] = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            out.append(p)
    out.sort()
    return out


def verify_dataset(
    dataset: str | Path,
    *,
    expected_names: list[str] | None = None,
    check_pairs: bool = True,
) -> VerifyReport:
    dataset_path = Path(dataset)
    dataset_yaml = _resolve_dataset_yaml(dataset_path)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    cfg = load_yaml(dataset_yaml)
    root = _resolve_root(dataset_yaml, cfg)

    names = _as_names(cfg.get("names"))
    nc = int(cfg.get("nc", len(names) or 0) or 0)
    if names and nc != len(names):
        nc = len(names)

    warnings: list[str] = []
    errors: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    def err(msg: str) -> None:
        errors.append(msg)

    if expected_names is not None:
        exp = [str(x) for x in expected_names]
        if names != exp:
            err(f"Class names mismatch: dataset has names={names}, expected={exp}")

    if nc <= 0 or not names:
        err("Invalid dataset.yaml: missing/empty `names` or `nc`.")

    images_total = 0
    empty_labels_total = 0
    boxes_total = 0
    boxes_per_cls: dict[int, int] = {}

    for subset in ("train", "val", "test"):
        images_rel = cfg.get(subset)
        if not images_rel:
            continue
        images_dir = (root / str(images_rel)).resolve()
        if not images_dir.exists():
            err(f"Missing {subset} images dir: {images_dir}")
            continue
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.exists():
            err(f"Missing {subset} labels dir: {labels_dir}")
            continue

        images = _iter_images(images_dir)
        images_total += len(images)

        image_stems = {p.stem for p in images}
        for txt in labels_dir.glob("*.txt"):
            if txt.stem not in image_stems:
                warn(f"{subset}: label without image: {txt.name}")

        for img in images:
            lbl = labels_dir / f"{img.stem}.txt"
            if not lbl.exists():
                err(f"{subset}: missing label for image: {img.name}")
                continue
            text = lbl.read_text().strip()
            if not text:
                empty_labels_total += 1
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    err(f"{subset}: {lbl.name}:{line_no}: expected 5 fields, got {len(parts)}")
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:]]
                except Exception:
                    err(f"{subset}: {lbl.name}:{line_no}: invalid YOLO line: {line!r}")
                    continue

                if cls_id < 0 or (nc and cls_id >= nc):
                    err(f"{subset}: {lbl.name}:{line_no}: class id {cls_id} out of range [0,{nc - 1}]")
                for name_val, v in (("xc", xc), ("yc", yc), ("bw", bw), ("bh", bh)):
                    if v < 0.0 or v > 1.0:
                        err(f"{subset}: {lbl.name}:{line_no}: {name_val}={v} out of [0,1]")
                if bw <= 0.0 or bh <= 0.0:
                    err(f"{subset}: {lbl.name}:{line_no}: invalid box size bw={bw} bh={bh}")

                boxes_total += 1
                boxes_per_cls[cls_id] = boxes_per_cls.get(cls_id, 0) + 1

    if check_pairs:
        meta_path = root / "meta.json"
        if not meta_path.exists():
            warn("meta.json not found; skipping multi-view pair consistency checks.")
        else:
            meta = json.loads(meta_path.read_text())
            groups = meta.get("groups")
            if not isinstance(groups, dict) or not groups:
                warn("meta.json has no `groups`; pair consistency is not guaranteed (use `groups:` in labeling config).")
            else:
                prefix_to_group_cam: dict[str, tuple[str, str]] = {}
                expected_cams: dict[str, set[str]] = {}
                for group_name, cams_obj in groups.items():
                    if not isinstance(cams_obj, dict) or not cams_obj:
                        continue
                    cams = [str(c) for c in cams_obj.keys()]
                    expected_cams[str(group_name)] = set(cams)
                    for cam in cams:
                        prefix = f"{group_name}_{cam}_f"
                        if prefix in prefix_to_group_cam:
                            err(f"Duplicate prefix '{prefix}' in groups; choose unique group/cam names.")
                        prefix_to_group_cam[prefix] = (str(group_name), cam)

                subset_for_key: dict[tuple[str, int], str] = {}
                cams_for_key: dict[tuple[str, int], set[str]] = {}

                for subset in ("train", "val", "test"):
                    images_rel = cfg.get(subset)
                    if not images_rel:
                        continue
                    images_dir = (root / str(images_rel)).resolve()
                    for img in _iter_images(images_dir):
                        stem = img.stem
                        if "_f" not in stem:
                            continue
                        left, idx_str = stem.rsplit("_f", 1)
                        prefix = f"{left}_f"
                        hit = prefix_to_group_cam.get(prefix)
                        if hit is None:
                            continue
                        if not idx_str.isdigit():
                            warn(f"{subset}: unexpected frame index suffix in '{stem}'")
                            continue
                        idx = int(idx_str)
                        group, cam = hit
                        key = (group, idx)
                        prev_subset = subset_for_key.get(key)
                        if prev_subset is None:
                            subset_for_key[key] = subset
                        elif prev_subset != subset:
                            err(f"Frame {group} f{idx:06d} appears in multiple subsets: {prev_subset}, {subset}")
                        cams_for_key.setdefault(key, set()).add(cam)

                for (group, idx), cams in cams_for_key.items():
                    exp = expected_cams.get(group)
                    if not exp:
                        continue
                    missing = exp - cams
                    if missing:
                        err(f"Missing cams for {group} f{idx:06d}: {sorted(missing)}")

    boxes_per_class: dict[str, int] = {}
    if names:
        for cls_id, count in boxes_per_cls.items():
            label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            boxes_per_class[label] = int(count)

    return VerifyReport(
        dataset_yaml=dataset_yaml,
        root=root,
        names=names,
        images=images_total,
        empty_labels=empty_labels_total,
        boxes_total=boxes_total,
        boxes_per_class=boxes_per_class,
        warnings=warnings,
        errors=errors,
    )
