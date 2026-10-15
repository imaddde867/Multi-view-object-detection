"""Lockstep frame iteration over a group of camera captures.

Both pipelines walked their camera captures with the same skeleton — read
every camera, apply a stride, stop at a cap, stop when any camera ends —
and differed only in the alignment policy. This module owns the skeleton;
``align`` selects the policy:

- ``"naive"``: read every capture each raw frame (labeling).
- ``"timestamp"``: drive on the sync camera's timestamps; other captures
  advance to their first frame at or after the sync timestamp (run).

Raw indices are yielded, not kept counts: with stride=1 they are the same,
and with stride>1 raw indices keep association aging in real frame units.
"""

from __future__ import annotations

from typing import Any, Iterator

import cv2


def iter_group_frames(
    caps: dict[str, Any],
    *,
    stride: int = 1,
    max_frames: int | None = None,
    align: str = "timestamp",
    sync_cam: str | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if not caps:
        return
    cam_names = list(caps.keys())
    if align == "naive":
        yield from _iter_naive(caps, cam_names, stride=stride, max_frames=max_frames)
    elif align == "timestamp":
        yield from _iter_timestamp(caps, cam_names, stride=stride, max_frames=max_frames, sync_cam=sync_cam)
    else:
        raise ValueError(f"Unknown align policy: {align!r} (expected 'naive' or 'timestamp')")


def _cap_exhausted(max_frames: int | None, kept: int) -> bool:
    # 0/None means unlimited, matching both pipelines' old behavior
    return bool(max_frames) and kept >= int(max_frames)


def _iter_naive(caps, cam_names, *, stride, max_frames):
    raw_idx = -1
    kept = 0
    while True:
        raw_idx += 1
        frames: dict[str, Any] = {}
        ok = True
        for cam in cam_names:
            ret, frame = caps[cam].read()
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
        if _cap_exhausted(max_frames, kept):
            break


def _iter_timestamp(caps, cam_names, *, stride, max_frames, sync_cam):
    if sync_cam is None or sync_cam not in caps:
        sync_cam = cam_names[0]
    raw_idx = -1
    kept = 0
    while True:
        ret, sync_frame = caps[sync_cam].read()
        if not ret:
            break
        raw_idx += 1
        if raw_idx % stride != 0:
            continue
        sync_ts = float(caps[sync_cam].get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        frames: dict[str, Any] = {sync_cam: sync_frame}
        ok = True
        for cam in cam_names:
            if cam == sync_cam:
                continue
            while True:
                ret2, frame2 = caps[cam].read()
                if not ret2:
                    ok = False
                    break
                ts = float(caps[cam].get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if ts >= sync_ts:
                    frames[cam] = frame2
                    break
            if not ok:
                break
        if not ok:
            break
        yield raw_idx, frames
        kept += 1
        if _cap_exhausted(max_frames, kept):
            break
