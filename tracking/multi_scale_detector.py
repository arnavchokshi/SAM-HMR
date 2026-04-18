"""Multi-scale YOLO detection ensemble.

Runs the same YOLO weights at multiple ``imgsz`` values per frame and
NMS-unions the boxes. Useful when subjects span a wide scale range
within a clip (BigTest: foreground + back-row dancers): a single
``imgsz`` always misses one band. The ensemble closes the gap without
re-training.

Phase 2 of the BigTest accuracy work uses this to reduce the
12 detector-undershoot frames remaining after Phase 0 (GT re-annotation)
and Phase 1 (proximity-gated long-gap merge).

The signature ``frame_bgr -> ndarray[N, 6]`` matches the detector hook
already used by ``eval/run_boxmot_tracker.py``, so plumbing this into a
BoxMOT tracker is a one-line drop-in.

Per-frame algorithm:
  1. Detect at each imgsz, classes=[0] (person).
  2. Concatenate all per-scale boxes.
  3. Run torchvision NMS at the supplied ``ensemble_iou`` (default 0.6).
     Score = max(conf) per merged box (the standard "max" reduction).

The ensemble is order-deterministic given the weights/seed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np


log = logging.getLogger(__name__)


def make_multi_scale_detector(
    weights: Path,
    *,
    imgsz_list: List[int],
    conf: float,
    iou: float,
    device: str,
    ensemble_iou: float = 0.6,
    classes: Optional[List[int]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a per-frame detector that runs YOLO at every ``imgsz`` in
    ``imgsz_list`` and NMS-unions the results.

    Output layout matches the BoxMOT detector contract:
        ``[x1, y1, x2, y2, conf, cls]``

    When ``len(imgsz_list) == 1`` this is exactly equivalent to the
    single-scale detector path used by ``eval/run_boxmot_tracker.py``,
    so callers can always wire this in unconditionally.

    ``ensemble_iou`` is the NMS IoU threshold used to fuse cross-scale
    duplicates (default 0.6 — same person at two scales typically has
    IoU >= 0.6, while two genuinely different people sit well below).
    """
    if not imgsz_list:
        raise ValueError("imgsz_list must contain at least one imgsz")
    if classes is None:
        classes = [0]

    from ultralytics import YOLO
    import torch
    from torchvision.ops import nms

    model = YOLO(str(weights))
    sorted_imgsz = sorted({int(s) for s in imgsz_list})
    log.info("multi-scale detector: weights=%s imgsz=%s conf=%.3f iou=%.3f "
             "ensemble_iou=%.2f device=%s classes=%s",
             weights, sorted_imgsz, conf, iou, ensemble_iou, device, classes)

    def detect(frame_bgr: np.ndarray) -> np.ndarray:
        all_xyxy: List[np.ndarray] = []
        all_conf: List[np.ndarray] = []
        all_cls: List[np.ndarray] = []
        for imgsz in sorted_imgsz:
            results = model.predict(
                frame_bgr, imgsz=int(imgsz), conf=conf, iou=iou,
                device=device, verbose=False, classes=classes,
            )
            if not results:
                continue
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            all_xyxy.append(boxes.xyxy.cpu().numpy().astype(np.float32))
            all_conf.append(boxes.conf.cpu().numpy().astype(np.float32))
            all_cls.append(boxes.cls.cpu().numpy().astype(np.float32))

        if not all_xyxy:
            return np.zeros((0, 6), dtype=np.float32)

        xyxy = np.concatenate(all_xyxy, axis=0)
        conf_arr = np.concatenate(all_conf, axis=0)
        cls_arr = np.concatenate(all_cls, axis=0)

        if len(sorted_imgsz) == 1:
            out = np.concatenate(
                [xyxy, conf_arr[:, None], cls_arr[:, None]], axis=1,
            ).astype(np.float32)
            return out

        boxes_t = torch.from_numpy(xyxy)
        scores_t = torch.from_numpy(conf_arr)
        keep = nms(boxes_t, scores_t, float(ensemble_iou)).cpu().numpy()
        xyxy = xyxy[keep]
        conf_arr = conf_arr[keep]
        cls_arr = cls_arr[keep]

        out = np.concatenate(
            [xyxy, conf_arr[:, None], cls_arr[:, None]], axis=1,
        ).astype(np.float32)
        return out

    return detect
