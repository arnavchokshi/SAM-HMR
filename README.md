# Dance multi-person tracking — YOLO ensemble + DeepOcSort

This repository ships **one** tracking pipeline, locked to the
configuration that won our 8-clip canonical scoreboard:

```
video
  │
  ▼
[1] YOLO multi-scale ensemble       weights/best.pt   (dancer-fine-tuned)
  │                                  imgsz = [768, 1024], cross-scale NMS
  │
  ▼
[2] BoxMOT DeepOcSort               OSNet x0.25 ReID
  │                                  + cholesky-jitter Kalman patch
  │
  ▼
[3] tracking/postprocess.py         conf/length prune · interp ·
                                    short-gap IoU+ReID merge ·
                                    long-gap proximity merge · smoothing
  │
  ▼
per-frame, per-track 2D boxes (and overlay MP4s if requested)
```

**Headline metric:** mean IDF1 = **0.949** across the 6 non-leaked
ground-truth clips (`BigTest, gymTest, adiTest, easyTest, loveTest,
shorterTest`).

The only docs you need:

- [`docs/WINNING_PIPELINE_CONFIGURATION.md`](docs/WINNING_PIPELINE_CONFIGURATION.md)
  — the locked configuration and how to run it.
- [`docs/EXPERIMENTS_LOG.md`](docs/EXPERIMENTS_LOG.md) — every other
  tracker / detector / post-process variant we tested, the scores, and
  why each was rejected.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

OSNet ReID weights are auto-downloaded by BoxMOT on first use; YOLO
weights live at `weights/best.pt` (committed). On Apple Silicon use
`--device mps`; on NVIDIA Hopper use `--device cuda:0`.

---

## Run the canonical 8-clip demo

```bash
python scripts/run_winner_stack_demo.py --device auto
```

Produces under `runs/winner_stack_demo/`:

- `_cache/<clip>/*.pkl` — per-clip `FrameDetections` cache.
- `overlays/<clip>_tracking_overlay.mp4` — per-clip ID overlay video.
- `timings.json` — per-clip wall time.

Restrict to a subset with `--clips BigTest loveTest`. Skip the
overlay-rendering pass with `--skip-render`.

---

## Run on a single ad-hoc video

```bash
python eval/run_boxmot_tracker.py \
    --tracker DeepOcSort \
    --weights weights/best.pt \
    --imgsz-ensemble 768 1024 --ensemble-iou 0.6 \
    --conf 0.31 --iou 0.70 \
    --device auto \
    --output runs/my_run \
    --extra-clip my_clip=/abs/path/to/video.mov \
    --clips my_clip
```

Then render an overlay or score against MOT GT — see
`docs/WINNING_PIPELINE_CONFIGURATION.md` §2.3 / §2.4.

---

## Repository layout (post-cleanup)

| Path | Purpose |
|------|---------|
| `tracking/deepocsort_runner.py` | The shipped tracker: BoxMOT DeepOcSort + cholesky-jitter Kalman patch + RawTrack output. |
| `tracking/multi_scale_detector.py` | YOLO `[768, 1024]` ensemble with cross-scale NMS union. |
| `tracking/postprocess.py` | `RawTrack[]` → cleaned `Track[]` (prune, interp, ID merge, smooth). |
| `prune_tracks.py` | `FrameDetections` legacy dataclass used by the eval cache. |
| `eval/run_boxmot_tracker.py` | Single-clip BoxMOT runner that produces the eval cache. |
| `eval/render_overlay_videos.py` | Renders overlay MP4s from a cache, applies the winner post-process. |
| `eval/regt/score_8clip.py` | IDF1 / MOTA / IDsw / `count_exact` on the 8-clip set with all post-process presets. |
| `eval/overfit/_common.py` | Clip registry + `PostCfg` + scoring helpers used by the regt scorer. |
| `scripts/run_winner_stack_demo.py` | One-shot driver: tracking + overlays for the canonical clips. |
| `scripts/fetch_reid.sh` | Downloads OSNet x0.25 ReID weights (offline-friendly fallback to BoxMOT auto-download). |
| `scripts/Dockerfile.trackers` | Reference container for reproducing the BoxMOT environment. |
| `weights/best.pt` | Dancer-fine-tuned YOLO26s — the locked detector. |
| `docs/WINNING_PIPELINE_CONFIGURATION.md` | Locked config + how-to-run. |
| `docs/EXPERIMENTS_LOG.md` | Every alternative we tried, scoreboards, kill criteria. |

Anything you do not see in this table was deleted in the cleanup. See
`docs/EXPERIMENTS_LOG.md` for what each removed module *was* and why it
did not make the cut.

---

## Hard rules

- Never score against `mirrorTest` for ranking decisions — leaked into
  YOLO fine-tune.
- Always report IDF1 alongside `count_exact` — `count_exact` is a
  trailing indicator.
- Never tune a global knob to lift one specific clip — see
  `docs/EXPERIMENTS_LOG.md` §3.4.
- Never touch `weights/best.pt` without re-running the full 8-clip
  scoreboard.
