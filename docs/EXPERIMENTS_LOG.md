# Experiments log — one consolidated history

> **TL;DR — the winner.** YOLO multi-scale ensemble (`imgsz=[768, 1024]`,
> `ensemble_iou=0.6`, `conf=0.31`) → **BoxMOT DeepOcSort** with OSNet
> x0.25 ReID → post-processing `min_total_frames=60`, proximity-merge
> `pose_max_center_dist=150 px / pose_max_gap=120 frames`. Mean IDF1 =
> **0.949** across the 6 non-leaked GT clips.
>
> All runtime parameters live in
> [`docs/WINNING_PIPELINE_CONFIGURATION.md`](WINNING_PIPELINE_CONFIGURATION.md).
> This document is the *history* — every alternative we evaluated and
> why it was rejected. New experiments should be appended at the bottom.

---

## 1. The 8-clip canonical set

| Clip | Dancers | GT? | Used to rank? | Notes |
|------|--------:|:---:|:--:|------|
| `BigTest` | 14 | yes | yes | 14 same-uniform dancers — hardest reID test. |
| `mirrorTest` | 9 | yes | **no — leaked** | In YOLO fine-tune. Sanity check only. |
| `gymTest` | 7 | yes | yes |  |
| `adiTest` | 5 | yes | yes | 188-frame cap (background YOLO false-positives after that). |
| `easyTest` | 6 | yes | yes |  |
| `2pplTest` | 2 | no | no — `count_exact` only | Constant 2 dancers. |
| `loveTest` | 15 | yes | yes | Free-form, varied outfits, **bottleneck clip**. |
| `shorterTest` | 9 | yes | yes |  |

`mean6nl` is the IDF1 mean across the 6 ranking clips
(`BigTest, gymTest, adiTest, easyTest, loveTest, shorterTest`).

---

## 2. Final scoreboard (representative configs)

All numbers from the same eval harness (`eval/regt/score_8clip.py`) on
the same cached YOLO detections.

### 2.1 Trackers (locked-in detector ensemble + winner post-process)

| Tracker (BoxMOT 10.0.52) | mean6nl IDF1 | mean6nl MOTA | Notes |
|--------------------------|------:|------:|------|
| **DeepOcSort + OSNet x0.25** | **0.949** | **0.909** | **Shipped default** |
| DeepOcSort + OSNet x1.0 | 0.948 | 0.908 | No gain, 4× slower embed |
| DeepOcSort + OSNet AIN x1.0 | 0.946 | 0.906 | Same |
| BotSort + OSNet x0.25 | 0.937 | 0.895 | -1.2 pp IDF1 |
| OcSort (no ReID) | 0.927 | 0.881 | -2.2 pp; loveTest at 0.71 |
| HybridSort | 0.921 | 0.870 | -2.8 pp |
| ByteTrack | 0.901 | 0.862 | -4.8 pp; expected (no ReID, no app cues) |
| StrongSort | 0.918 | 0.872 |  |
| **CAMELTrack (DanceTrack ckpt)** | 0.872 | 0.821 | -7.7 pp; see §3.4 |

### 2.2 Detector ensemble scales

| `imgsz_ensemble` | BigTest IDF1 | mean6nl IDF1 | Notes |
|------------------|------:|------:|------|
| `[768]` only | 0.961 | 0.937 | Single scale baseline |
| `[1024]` only | 0.951 | 0.929 | Misses small dancers |
| **`[768, 1024]`** | **0.998** | **0.949** | **Shipped** |
| `[768, 1024, 1280]` | 0.854 | 0.928 | 1280 introduces YOLO false-positive splits |
| `[640, 768, 1024]` | 0.985 | 0.940 | No gain over 768+1024, 1.5× slower |

### 2.3 Post-processing presets (DeepOcSort detections)

(Encoded in `eval/regt/score_8clip.py::POSTS`.)

| Preset | `mtf` | `prox` | `gap` | `pose_cos` | mean6nl IDF1 | Notes |
|--------|-----:|-----:|-----:|-----:|------:|------|
| `none` (no merge) | 30 | – | – | 0 | 0.918 | Baseline |
| `legacy_short` | 30 | 80 | 30 | 0 | 0.931 |  |
| `winner_mtf60_prox150_pose80` | 60 | 150 | 120 | 0.80 | 0.943 | Adds pose-cos |
| **`winner_mtf60_prox150`** | **60** | **150** | **120** | **0** | **0.949** | **Shipped — pose-cos OFF** |
| `aggressive_long` | 90 | 250 | 240 | 0 | 0.941 | Over-merges loveTest |
| `mtf30_prox150` | 30 | 150 | 120 | 0 | 0.945 | mtf=60 gives small lift |

`pose_cos_thresh > 0` regresses loveTest by ~2 pp (over-merges
visually-similar dancers). It stays OFF.

---

## 3. Things tried and why they were dropped

### 3.1 CAMELTrack (learned association)

- **What:** Drop-in replacement for the heuristic association in BotSort
  with the CAMELTrack DanceTrack-pretrained transformer.
- **Result:** mean6nl IDF1 = 0.872 vs DeepOcSort 0.949 ( **-7.7 pp** ).
  Worst on `BigTest` (uniform dancers) where it confused identities
  reID would have caught. Hyperparameter sweeps (gating thresholds,
  reID weight, BPBReID swap) moved it ≤ 0.5 pp.
- **Verdict:** Retired. The DanceTrack checkpoint does not transfer to
  our same-uniform crowded clips, and we have no labelled dance data
  to fine-tune. Not shipped.

### 3.2 Bigger ReID embedders (OSNet x1.0, OSNet AIN x1.0)

- **What:** Swap OSNet x0.25 → OSNet x1.0 / OSNet AIN x1.0.
- **Result:** ≤ 0.001 IDF1 swing in either direction; OSNet AIN x1.0 is
  **slightly worse** on `loveTest`.
- **Verdict:** Bottleneck is association, not embedding capacity.
  OSNet x0.25 ships (smaller and faster, no accuracy loss).

### 3.3 SAM 2.1 as tracker

- **What:** Replace box tracker with SAM 2.1 video predictor seeded by
  YOLO masks; derive boxes from masks.
- **Result:** Excellent on short isolated clips, blew up on the long
  same-uniform ones — mask propagation fused identities once two
  dancers occluded. mean6nl IDF1 ≈ 0.78 in the one full run.
- **Verdict:** Retired. Kept the post-processing pipeline because it
  was tracker-agnostic but removed the SAM2 tracker module entirely.

### 3.4 Per-clip / per-track oracle tuning

- **What:** Optuna sweeps over post-processing knobs per individual
  clip; per-track post-process picks.
- **Result:** Per-clip oracle (best knob set chosen *with knowledge of
  the GT for that clip*) tops out at mean6nl IDF1 = 0.948 across all
  trackers in our family — i.e. **the per-clip oracle is below our
  global winner**. There is nothing left to extract from the
  heuristic-tracker family without a different association method.
- **Verdict:** Retired. We explicitly stopped tuning per-clip.

### 3.5 Pose-cosine identity merge

- **What:** Use VitPose-25 keypoints to compute a cosine similarity
  between two short tracks before merging across an occlusion gap.
- **Result:** Helps `BigTest` slightly (+0.3 pp), regresses `loveTest`
  (-2 pp) by over-merging visually similar but distinct dancers.
  Net mean6nl: -0.6 pp.
- **Verdict:** Disabled (`pose_cos_thresh = 0`) but the code path is
  retained in `tracking/postprocess.py` behind a knob in case future
  GT supports re-enabling it.

### 3.6 Wider detector ensemble (`+1280`)

- **What:** Add a 1280-px scale to the YOLO ensemble for higher recall
  on small / distant dancers.
- **Result:** `BigTest` IDF1 collapses 0.998 → 0.854 because YOLO
  splits each large foreground dancer into multiple boxes at 1280 px,
  defeating cross-scale NMS.
- **Verdict:** Retired. Ensemble is locked at exactly `[768, 1024]`.

### 3.7 PromptHMR full 3D pipeline

- **What:** End-to-end SMPL-X reconstruction with PHMR image head +
  PHMR-Vid + DROID-SLAM world alignment + Meshcapade `.mcs` export.
- **Result:** Worked, but the surface area was huge (CUDA sm_90 only,
  many license-gated weights, fragile aarch64 wheel matrix), and we
  determined the 3D head can be re-attached on top of stable tracking
  later.
- **Verdict:** Retired from this repo. This repo is now the *tracking*
  half only — input video → cleaned per-track 2D boxes. The 3D head is
  a future, separate concern.

### 3.8 Limb-rescue / hand-outlier rejection / VitPose hand pass

- **What:** Per-frame keypoint correction passes (limb rescue, hand
  outlier rejection, HaMeR hand recovery).
- **Result:** Useful for downstream 3D, irrelevant to tracker ranking
  and only made sense as part of the PHMR stack.
- **Verdict:** Removed with the PHMR stack.

---

## 4. What this repo *does* ship

- `tracking/deepocsort_runner.py` — BoxMOT DeepOcSort wrapped with the
  cholesky-jitter Kalman patch.
- `tracking/multi_scale_detector.py` — YOLO `[768, 1024]` ensemble with
  cross-scale NMS union.
- `tracking/postprocess.py` — pruning, interp, ID-merge, smoothing.
- `eval/run_boxmot_tracker.py` — single-clip BoxMOT runner that
  produces the cache layout the rest of the eval consumes.
- `eval/render_overlay_videos.py` — overlay MP4 renderer with the
  winner post-process applied.
- `eval/regt/score_8clip.py` — IDF1 / MOTA / IDsw / `count_exact` on
  the 8-clip canonical set with all post-process presets.
- `eval/overfit/_common.py` — clip registry + `PostCfg` + scoring used
  by the regt scorer.
- `scripts/run_winner_stack_demo.py` — one-shot driver: tracking +
  overlays for the canonical clips with the winner config locked.

Anything you do not see in this list was deleted in the cleanup —
including all CAMELTrack, SAM2-tracker, tracker A/B/C, sweep, oracle,
and PHMR-pipeline modules.

---

## 5. Future experiments (open)

Things that *could* lift `loveTest` past 0.80 IDF1, in order of
expected payoff:

1. **Custom ReID head** trained on dance footage (the only thing the
   per-clip oracle ceiling at 0.948 *cannot* reach).
2. **Group-/synchrony-aware association** — exploit the fact that
   dancers in the same routine move correlated.
3. **SAM 2.1 mask cascade as a *post-tracker* IoU floor** rather than a
   primary tracker.

If any of these are tried, append a §3.x entry above with the same
shape (What / Result / Verdict).
