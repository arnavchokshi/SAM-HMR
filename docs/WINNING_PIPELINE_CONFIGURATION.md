# Winning pipeline — configuration & how to run

> **Single shipped pipeline:** YOLO multi-scale ensemble + **BoxMOT
> DeepOcSort** + post-processing (`mtf=60`, `prox=150 px`, `gap=120 frames`).
>
> **Headline:** mean IDF1 = **0.949** across the 6 non-leaked GT clips.

The full scoreboard, every alternative tested, and the rationale for each
choice live in [`docs/EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md). This doc
is just the working configuration and how to run it.

---

## 1. Configuration (locked)

### 1.1 Detector

| Field | Value |
|------|------|
| YOLO weights | `weights/best.pt` (dancer-fine-tuned YOLO26s) |
| `imgsz_ensemble` | `[768, 1024]` — runs YOLO at both scales per frame |
| `ensemble_iou` | `0.6` — cross-scale NMS union |
| `conf` | `0.31` |
| `iou` | `0.70` |
| `classes` | `[0]` (person) |

Adding `1280` to the ensemble regresses BigTest IDF1 from 0.998 to 0.854,
so the ensemble stays at exactly two scales.

### 1.2 Tracker

| Field | Value |
|------|------|
| Library | BoxMOT (`boxmot >= 10.0.52`) |
| Class | **`DeepOcSort`** |
| ReID | `osnet_x0_25_msmt17.pt` (BoxMOT auto-downloads) |
| Device | `cuda:0` (NVIDIA) or `mps` (Apple Silicon) |

A cholesky-jitter patch is installed at runtime (`tracking/deepocsort_runner.py`
`install_kalman_jitter_patch`) — DeepOcSort's Kalman update occasionally
hits a non-SPD covariance otherwise.

Bigger ReID embedders (OSNet x1.0, OSNet AIN x1.0) were tried and did not
move the needle — the bottleneck is the association method, not the
embedding quality. The 4× smaller OSNet x0.25 also slightly **wins**
loveTest, so that stays the default.

### 1.3 Post-processing

| Knob | Value |
|------|------|
| `min_total_frames` | `60` |
| `min_conf` | `0.38` |
| `max_gap_interp` | `12` |
| `id_merge.iou_thresh` / `max_gap_frames` | `0.5` / `8` (short-gap IoU merge) |
| `id_merge.osnet_cos_thresh` | `0.7` |
| `pose_max_center_dist` | `150 px` (long-gap proximity merge) |
| `pose_max_gap` | `120 frames` |
| `pose_cos_thresh` | `0` (pose-cosine merge OFF — regressed loveTest by ~2pp) |
| `smoothing.medfilt_window` | `11` |
| `smoothing.gaussian_sigma` | `3.0` |

These knobs are encoded in `eval/overfit/_common.py::PostCfg` and the
matching variant in `eval/regt/score_8clip.py::POSTS["winner_mtf60_prox150"]`.

---

## 2. How to run

### 2.1 The whole 8-clip canonical set (tracking + overlays in one shot)

```bash
python scripts/run_winner_stack_demo.py --device auto
# auto picks cuda:0 / mps / cpu in that order
```

Output:

```
runs/winner_stack_demo/
  _cache/<clip>/*.pkl                  # FrameDetections cache per clip
  overlays/<clip>_tracking_overlay.mp4 # ID overlay video
  timings.json                         # per-clip wall time
```

Restrict to a subset:

```bash
python scripts/run_winner_stack_demo.py --clips BigTest loveTest --device mps
```

### 2.2 A single ad-hoc video

```bash
python eval/run_boxmot_tracker.py \
  --tracker DeepOcSort \
  --weights weights/best.pt \
  --imgsz-ensemble 768 1024 --ensemble-iou 0.6 \
  --conf 0.31 --iou 0.70 \
  --device mps \
  --output runs/my_run \
  --extra-clip my_clip=/abs/path/to/video.mov \
  --clips my_clip
```

### 2.3 Score an existing cache against MOT ground truth

```bash
python eval/regt/score_8clip.py \
  --variants <subdir-under-runs/bestofboth_gh200> \
  --post winner_mtf60_prox150
```

`--post all` enumerates every preset in
`eval/regt/score_8clip.py::POSTS` so you can see the sensitivity grid in
one pass.

### 2.4 Render overlays from an existing cache

```bash
python eval/render_overlay_videos.py \
  --cache-root runs/winner_stack_demo/_cache \
  --output runs/winner_stack_demo/overlays \
  --min-total-frames 60 --min-conf 0.38 \
  --pose-max-center-dist 150 --pose-max-gap 120 \
  --config-label "DeepOcSort | ens 768+1024 | mtf=60 prox=150 gap=120"
```

---

## 3. 8-clip headline scoreboard (DeepOcSort + winner post)

| Clip          | IDF1  | MOTA  | count_exact | tracks / GT |
|---------------|------:|------:|------------:|:-----------:|
| BigTest       | 0.998 | 0.996 | 0.971 | 14 / 14 |
| mirrorTest    | 0.986 | 0.972 | 1.000 | 9 / 9 *(LEAKED — not used for ranking)* |
| gymTest       | 0.974 | 0.980 | 0.861 | 7 / 7 |
| adiTest       | 1.000 | 1.000 | 1.000 | 5 / 5 |
| easyTest      | 1.000 | 1.000 | 1.000 | 6 / 6 |
| 2pplTest      |  n/a  |  n/a  | 1.000 | 2 / 2 *(no GT — constant 2)* |
| loveTest      | 0.804 | 0.638 | 0.637 | 16 / 15 — **bottleneck** |
| shorterTest   | 0.919 | 0.838 | 0.974 | 9 / 9 |
| **mean6nl**   | **0.949** | 0.909 | 0.907 |  |

`mean6nl` = mean across `gymTest, adiTest, BigTest, easyTest, loveTest,
shorterTest` (excludes leaked mirrorTest and the no-GT 2pplTest).

`loveTest` (15 free-form dancers, varied outfits) is the only clip that
falls below the 0.95 IDF1 target. Per-clip oracle ceiling across all
heuristic trackers we tested is **0.948 mean6nl** — i.e. there is no
heuristic tracker hyperparameter that pushes loveTest above ~0.80 IDF1.
The bottleneck is the association algorithm itself; see
[`docs/EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md) §3.

---

## 4. Hardware notes

- **Apple Silicon:** use `--device mps`.
- **NVIDIA Hopper / GH200:** use `--device cuda:0`. CUDA caches are not
  bit-identical to MPS for the same weights — score on the platform you
  run.
- BoxMOT's OSNet ReID is auto-downloaded to its cache on first use; no
  explicit `--reid-weights` argument required.

---

## 5. Hard rules (do not break)

- Never score against `mirrorTest` for ranking decisions — leaked into
  YOLO fine-tune.
- Always report IDF1 alongside `count_exact`. `count_exact` is a trailing
  indicator; IDF1 is the real metric.
- Never tune a global knob to lift one specific clip — this repo
  explicitly walked back from per-clip optimisation. See
  [`docs/EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md) §3 for the kill-criteria
  history.
- Never touch `weights/best.pt` without re-validating the full 8-clip
  scoreboard.
