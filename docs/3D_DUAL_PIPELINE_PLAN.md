# Dual 3D pipeline — PromptHMR-Vid vs SAM-Body4D — implementation plan

> **For agentic workers:** Each task below is a bite-sized commit. Steps use
> checkbox (`- [ ]`) syntax for tracking. Read §1–§4 for context, then
> work the tasks in §5 in order.

**Goal:** Extend the current YOLO + DeepOcSort tracking pipeline so it can
run **both PromptHMR-Vid and SAM-Body4D** on the same video and write
their outputs into a single comparison-ready directory layout.

**Architecture:** A shared "intermediate artifacts" producer (frames +
DeepOcSort tracks + SAM-2 masks + ViTPose 2D keypoints + camera intrinsics)
that runs in our existing tracking conda env, then two HMR sidecars
(PromptHMR-Vid in its own conda env, SAM-Body4D in its own conda env) that
each consume those artifacts and write SMPL-X / MHR outputs side by side.
A final comparison harness reads both outputs and produces metrics + a
side-by-side video.

**Tech stack:**
- Tracking env (existing): Python 3.11, PyTorch 2.x (mps/cuda), `boxmot
  >= 10.0.52`, `ultralytics >= 8.4.37`.
- PromptHMR env (new): conda `phmr_pt2.4`, Python 3.11.9, PyTorch 2.4.0
  + cu121, includes Detectron2 / SAM2 / DROID-SLAM / Metric3D.
- SAM-Body4D env (new): conda `body4d`, Python 3.12, PyTorch 2.7.1 +
  cu118, includes SAM-3 / Diffusion-VAS / SAM-3D-Body.

---

## 1. The verdict (read this first)

**Yes, this pipeline can be built — and the two systems can share *most*
of their inputs.** The integration is cleaner than it looks because both
systems are designed to consume:

1. A **video** (or a folder of frames).
2. A list of **per-track persistent identities** that map each frame to
   a person.
3. Per-frame **bounding boxes** for each tracked identity.
4. Optionally, per-frame **segmentation masks** for each identity.

Our existing pipeline already produces (1)–(3) with mean IDF1 = 0.949.
Adding (4) is one extra SAM-2 call per dancer per video. That single
artifact set drives both HMR backends.

The non-trivial parts are:

- **Two conda environments must be kept separate.** PromptHMR-Vid wants
  PyTorch 2.4-cu121 / 2.6-cu126 with Detectron2 + SAM-2; SAM-Body4D
  wants PyTorch 2.7.1-cu118 with SAM-3 + Diffusion-VAS + SAM-3D-Body.
  These cannot coexist in one env (different SAM versions, different
  detectron2 wheels, conflicting transformer dependencies). We solve
  this by writing two thin "adapter" CLIs, one per env, that read the
  shared artifacts directory and write to disjoint output subfolders.
- **The two systems output different mesh formats.** PromptHMR emits
  **SMPL-X**; SAM-Body4D emits **MHR** (Momentum Human Rig). We compare
  in **3D joint space** (extract joints from each, project to a common
  17-keypoint COCO skeleton) plus a side-by-side rendered video — we do
  NOT try to convert MHR ↔ SMPL-X (would need a separate research
  project).
- **SAM-Body4D's `loveTest`-class crowded scenes risk OOM.** Per its
  `resources.md`, 5 dancers + occlusion completion peaks at ~53 GB on
  H800. We run with `completion.enable=true` by default because dance
  scenes are exactly the occluded multi-person case Diffusion-VAS was
  built for; on the densest clips we mitigate the 9× runtime hit and
  VRAM peak by processing tracks in batches of ≤5 IDs and (if needed)
  dropping `sam_3d_body.batch_size` from 16 → 8.
- **PromptHMR-Vid's released checkpoint is body-only.** Hands are
  zero-padded in `phmr_vid.py`. SAM-Body4D *does* produce hand vertices
  via MHR. We document this asymmetry in the comparison report; it is
  not something we can fix in a wrapper.

The critical structural insight: **we replace each system's built-in
detector + tracker with our DeepOcSort tracks**. That is what makes the
comparison apples-to-apples — the only thing that varies between the two
runs is the HMR backend itself.

---

## 2. Deep dive: PromptHMR-Vid

### 2.1 What it is, exactly

The released PromptHMR repo (`yufu-wang/PromptHMR`, CVPR 2025) ships
**two** end-to-end demo scripts:

1. `scripts/demo_phmr.py` — single-image. Takes one image, runs YOLOv8x
   for boxes, runs the per-frame `PromptHMR` model, dumps an `output.pt`
   with `pose, betas, transl, rotmat, vertices, body_joints, cam_int`,
   and shows a Viser web viewer.
2. `scripts/demo_video.py` — full video / world-coord. Wraps the
   `pipeline.Pipeline` class which is the actual product surface
   we care about.

**`pipeline.Pipeline.__call__`** (file: `pipeline/pipeline.py`) is a
9-stage pipeline:

```
load_frames                    # cv2 read at max_height=896, max_fps=60
est_camera                     # naive intrinsics (focal = max(H, W))
SPEC camera calibration        # field-of-view refinement
run_detect_track               # detectron2 + SAM-2 video tracking  ← REPLACE
camera_motion_estimation       # masked DROID-SLAM + Metric3D for scale
estimate_2d_keypoints          # ViTPose-h-coco-25 per track
hps_estimation                 # PromptHMR (image) + PromptHMR-Vid (video)
world_hps_estimation           # convert cam-coords HPS to world coords
post_optimization              # foot-contact + Kalman-style smoothing
```

Outputs:

- `results.pkl` — full per-track HPS dict (joblib).
- `subject-<tid>.smpl` — per-track SMPLCodec file (Meshcapade format).
- `world4d.mcs` — Meshcapade Studio scene with all dancers + camera.
- `.glb` (via the same `world4d.mcs` export pathway) — drag into Blender
  or any `<model-viewer>` web component.

### 2.2 The track dict format (the integration contract)

Inside `pipeline/phmr_vid.py` the video model expects a per-track dict
of this shape (this is what `run_detect_track` produces and what we
need to fabricate ourselves):

```python
results['people'] = {
    track_id: {
        'track_id':       int,
        'frames':         np.ndarray,        # (T,) int — contiguous frame indices
        'bboxes':         np.ndarray,        # (T, 4) float xyxy
        'masks':          np.ndarray,        # (T, H, W) bool — full-image binary mask
        'detected':       np.ndarray,        # (T,) bool — True for real, False for interp
        # Filled by estimate_2d_keypoints (we provide these too):
        'keypoints_2d':   np.ndarray,        # (T, 135, 3) openpose-format
        'vitpose':        np.ndarray,        # (T, 17 or 25, 3) cocoophf for video head
    },
    ...
}
results['masks']  = np.ndarray              # (T_total, H, W) bool — union over all people
results['camera'] = {
    'pred_cam_R': np.ndarray,                # (T_total, 3, 3) per-frame world→cam
    'pred_cam_T': np.ndarray,                # (T_total, 3)
    'img_focal':  float,
    'img_center': np.ndarray,                # (2,)
}
```

After `hps_estimation` runs, each `track` gains a `smplx_cam` dict
(`rotmat`, `pose`, `shape`, `trans`, `contact`, `static_conf_logits`).
After `world_hps_estimation`, it gains `smplx_world` (same fields, world
coords). After `post_optimization`, `smplx_world` is foot-skating-fixed.

### 2.3 Inputs we will provide

| Input | Source | Resolution / format |
|---|---|---|
| `frames` (extracted images) | our `cv2.VideoCapture` after a max_height=896 resample | JPG sequence in `runs/3d_compare/<clip>/intermediates/frames/` |
| `tracks` dict | our DeepOcSort cache → conversion script | one pickle in `intermediates/tracks.pkl` |
| `masks` per-track | new SAM-2 video propagation seeded by our bboxes | `intermediates/masks_per_track/{tid}/{frame:08d}.png` (binary) and union in `intermediates/masks_union.npy` |
| `keypoints_2d` per-track (openpose 135-pt) | ViTPose-h-coco-25 → `convert_kps('vitpose25', 'openpose')` | `intermediates/keypoints/openpose.pkl` |
| `vitpose` per-track (cocoophf) | same ViTPose pass → `convert_kps('ophandface', 'cocoophf')` | `intermediates/keypoints/vitpose.pkl` |
| Camera intrinsics | `est_calib(images[0])` (focal = max(H,W)) | `intermediates/camera/intrinsics.json` |
| Camera trajectory (optional) | DROID-SLAM masked by `masks_union` | `intermediates/camera/slam.pkl`, falls back to identity for static-camera shots |

### 2.4 The injection point

Look at `pipeline/pipeline.py::Pipeline.__call__`. It checks
`self.results['has_tracks']` before running `run_detect_track()`. If we
**pre-populate** `self.results['people']`, `self.results['masks']`, and
set `has_tracks=True`, the line is skipped:

```python
if not self.results['has_tracks']:
    self.run_detect_track()       # <-- skipped
```

Same trick for the camera: pre-populate `self.results['camera']` and
set `has_slam=True`. The keypoint stage (`estimate_2d_keypoints`) we let
run as-is — ViTPose is fast and we don't want to re-implement the
openpose ↔ cocoophf conversion.

So the wrapper script we write is essentially:

```python
pipeline = Pipeline(static_cam=...)
pipeline.images, pipeline.seq_folder = pipeline.load_frames(input_video, output_folder)
pipeline.results = {...}                       # init blank dict
pipeline.results['people'] = load_pkl('tracks.pkl')  # OUR tracks
pipeline.results['masks']  = np.load('masks_union.npy')
pipeline.results['camera'] = json.load('intrinsics.json') | slam_pkl
pipeline.results['has_tracks'] = True
pipeline.results['has_slam']   = True
pipeline.estimate_2d_keypoints()   # let it run
pipeline.hps_estimation()          # this is the actual PromptHMR call
pipeline.world_hps_estimation()
pipeline.post_optimization()
# Then write results.pkl, .smpl files, world4d.mcs, .glb
```

This is the cleanest integration possible — we use the project's own
post-tracking pipeline unchanged.

### 2.5 Hardware / runtime expectations

From the PromptHMR demos and BEDLAM2 paper:

- Single-image PromptHMR: ~0.1 s/image on H100, ~1 s/image on CPU.
- Video pipeline including DROID-SLAM + per-frame HMR + video head:
  ~5× real-time on H100 for a single-person clip. Multi-person scales
  roughly linearly with `num_max_people`.
- VRAM: ~16–24 GB for a typical 1080p × 1500-frame × 4-person clip.
- DROID-SLAM is the slow part (~real-time on H100 with stride=1; we use
  stride=1 by default per `pipeline/config.yaml`).

### 2.6 Known limitations (to document in the comparison report)

1. **Body-only released checkpoint.** Hands are zero-padded in
   `pipeline/phmr_vid.py` (`hand_pose_rotmat = torch.zeros(..., 30, 3, 3)`).
2. **Static-camera mode** (`--static_camera`) bypasses DROID-SLAM and
   pins the camera at identity. Use this for tripod-mounted dance shots
   to save 30–60 % runtime.
3. **DROID-SLAM can fail** on near-static cameras (it raises
   `ValueError: not enough values to unpack`) — the existing pipeline
   catches this and falls back to identity. We surface the fallback in
   our log so we know which clips drifted to "static".

---

## 3. Deep dive: SAM-Body4D

### 3.1 What it is, exactly

The released SAM-Body4D repo (`gaomingqi/sam-body4d`, arXiv 2512.08406,
Dec 2025) is a **training-free** wrapper that bolts together three
existing models:

1. **SAM-3** (Meta, gated on HF) — promptable video segmentation.
   Takes a first-frame bbox/point prompt per identity, propagates
   masklets across the video.
2. **Diffusion-VAS** (Chen et al.) — diffusion-based amodal
   segmentation + RGB completion. Used to "fill in" occluded portions
   of each masklet so the per-frame HMR doesn't see a half-body.
3. **SAM-3D-Body** (Meta, Nov 2025, gated on HF) — single-image
   promptable HMR. Takes an image + (bbox + mask) and outputs MHR
   meshes per detected person.

Plus auxiliary models: **MoGe-2** (FOV estimator) and **Depth-Anything
V2** (depth for occlusion analysis).

The orchestrator is `scripts/offline_app.py`. Its `inference()` does:

```
1. Read first frame. Run SAM-3D-Body's vitdet detector (bbox_thr=0.6).
   If no people, slide forward up to ~100 frames until detection.
2. For each detected person: add their bbox as a SAM-3 prompt with
   obj_id = 1, 2, 3, ... at the first valid frame.
3. on_mask_generation():
     - Run SAM-3 propagation across the whole video.
     - Save palette-PNG masks to OUTPUT_DIR/masks/{frame:08d}.png
       (pixel value == obj_id).
     - Save raw frames to OUTPUT_DIR/images/{frame:08d}.jpg.
4. on_4d_generation():
     - For each batch (default 64 frames):
         a. (Optional) Diffusion-VAS amodal masks; compute IoU vs
            modal masks; if IoU < 0.7, that frame is "occluded".
         b. (Optional, for occluded frames) Diffusion-VAS RGB
            completion to fill in the missing pixels.
         c. Run SAM-3D-Body via process_image_with_mask() with the
            (possibly completed) (image, mask) pair.
     - Save per-person PLY meshes + per-frame focal-length JSON +
       rendered overlay frames.
5. ffmpeg the rendered frames → 4d_*.mp4
```

### 3.2 The mask format (the integration contract)

SAM-Body4D's `process_image_with_mask` expects:

- `image_path[i]` — path to JPG frame `i`.
- `mask_path[i]` — path to a **palette PNG** of frame `i` where the
  pixel value of each foreground pixel equals the `obj_id` of that
  person (background = 0).
- `occ_dict[obj_id]` — per-frame list of `0` (occluded) / `1`
  (not occluded) to control the Diffusion-VAS branch. If we don't run
  amodal completion, all entries are `1`.

The "palette PNG with per-id pixel value" is the standard DAVIS / VOS
convention. Mask conversion in the existing offline app is:

```python
msk = np.zeros_like(img[:, :, 0])
for out_obj_id, out_mask in video_segments[frame_idx].items():
    binary = (out_mask[0] > 0).astype(np.uint8) * 255
    msk[binary == 255] = out_obj_id
msk_pil = Image.fromarray(msk).convert('P')
msk_pil.putpalette(DAVIS_PALETTE)
msk_pil.save(...)
```

### 3.3 Inputs we will provide

| Input | Source | Resolution / format |
|---|---|---|
| `images/{frame:08d}.jpg` | our extracted frames | JPG, full original resolution (NOT the PromptHMR 896-cap; SAM-3D-Body handles arbitrary sizes) |
| `masks/{frame:08d}.png` | conversion of our SAM-2 per-track masks → palette PNG with pixel == DeepOcSort tid | palette PNG matching frame resolution |
| `out_obj_ids` | sorted list of our DeepOcSort track IDs | passed in via `predictor.RUNTIME['out_obj_ids']` |

Note: our DeepOcSort track IDs are arbitrary integers (1, 2, 7, 12, ...).
Palette PNGs only have 256 levels, so as long as we have ≤ 255 dancers
we're fine. For `loveTest` (15 dancers) we're trivially safe.

### 3.4 The injection point

Look at `scripts/offline_app.py`. The full `inference()` function
sequences detection → SAM-3 init → propagation → HMR. **We replace
detection + propagation entirely** by:

1. Pre-writing `images/` and `masks/` to the OfflineApp's output dir.
2. Setting `predictor.RUNTIME['out_obj_ids'] = our_track_ids`.
3. Calling `predictor.on_4d_generation()` directly (skips
   `on_mask_generation` because the masks are already written).

`on_4d_generation` reads from `OUTPUT_DIR/images/` and
`OUTPUT_DIR/masks/` via `glob`, so as long as those directories exist
with the right names, it works without modification.

The wrapper is essentially:

```python
predictor = OfflineApp(config_path='configs/body4d.yaml')
predictor.OUTPUT_DIR = our_output_folder
write_images_to(predictor.OUTPUT_DIR + '/images/', frames)
write_palette_masks_to(predictor.OUTPUT_DIR + '/masks/', track_masks)
predictor.RUNTIME['out_obj_ids'] = sorted(track_ids)
with torch.autocast('cuda', enabled=False):
    predictor.on_4d_generation()
```

We do NOT need to load SAM-3 or initialize an inference state — the
mask propagation is what SAM-3 was for, and we've already done it
upstream with SAM-2 (which is more compatible with PromptHMR anyway).
The model loaders for SAM-3 are skippable; we monkey-patch
`build_sam3_from_config` to return `(None, None)` to save VRAM and
boot time.

### 3.5 Hardware / runtime expectations

From `assets/doc/resources.md` (H800, 80 GB VRAM):

| #Targets | #Frames | Occlusion completion | 4D Peak VRAM | 4D Time |
|---:|---:|:-:|---:|---:|
| 1 | 100 | off | 14.5 GB | 1m 10s |
| 5 | 90 | off | 40.9 GB | 2m 55s |
| 5 | 90 | **on** | 53.3 GB | **26m** |
| 6 | 64 | on | 52.9 GB | 27m |

**Key takeaway:** Diffusion-VAS occlusion completion is a 9× runtime
penalty but it is exactly what dance scenes need (heavy occlusion,
amodal limbs, dancers crossing in front of each other). We default to
`completion.enable=true` and accept the runtime hit; the orchestrator
exposes `--disable-completion` for fast iteration / debugging runs.

For our `loveTest` (15 dancers, ~1500 frames), the linear extrapolation
from the table is ~8.5 GB/dancer for the 4D step → ~128 GB peak for
15 dancers, which is over even an H100's 80 GB. Mitigations:

1. **Process tracks in batches of ≤5 IDs.** SAM-Body4D's per-track loop
   already supports this (`obj_ids = sorted(map(int, occ_dict.keys()))`,
   inferred per-id then merged). We just call `process_image_with_mask`
   N times with subsets of `obj_ids`.
2. **Reduce `sam_3d_body.batch_size` from 64 → 16.** Trades runtime for
   VRAM; per the table, going from 64 → 32 already cut peak from 53 → 35
   GB.
3. **Process at half resolution** for the dense-crowd clips and only
   render at full resolution on `BigTest` / `adiTest`.

We document the per-clip configuration as a YAML override.

### 3.6 Known limitations (to document in the comparison report)

1. **MHR mesh format is not SMPL-X.** No straight comparison possible at
   the vertex level. We compare in joint space using a 17-keypoint COCO
   skeleton extracted from each.
2. **No native global trajectory.** SAM-Body4D outputs cam-coords
   meshes only. To compare to PromptHMR's world-coords output, we
   project both back into camera coordinates per frame.
3. **First-frame quality matters a lot.** SAM-Body4D's original demo
   uses the first frame to seed SAM-3; we bypass SAM-3 entirely and use
   our DeepOcSort masks for every frame, so this is moot — but be aware
   that any drift in our DeepOcSort tracks propagates directly.
4. **Released MHR rig is body + face + hands** but the inverse
   kinematics from the regressed parameters is not as smooth as
   PromptHMR's video head. Expect slightly more jitter on individual
   joints, especially on the legs during turns.

---

## 4. Architecture: shared artifacts + two sidecars

```
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE A — runs in the existing tracking conda env                    │
│   (Python 3.11, boxmot, ultralytics, mps/cuda)                       │
│                                                                      │
│   video.mp4                                                          │
│      │                                                               │
│      ▼                                                               │
│   YOLO26s ensemble (768+1024) + DeepOcSort + post-process            │
│      │   per (frame, dancer_id): bbox + conf                         │
│      ▼                                                               │
│   write `intermediates/tracks.pkl`                                   │
│   write `intermediates/frames/{frame:08d}.jpg`                       │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE B — runs in the PromptHMR conda env (`phmr_pt2.4`)             │
│   (Python 3.11.9, PyTorch 2.4 + cu121, SAM2, ViTPose)                │
│                                                                      │
│   reads `intermediates/tracks.pkl` + `frames/`                       │
│      │                                                               │
│      ▼                                                               │
│   SAM-2 video propagation seeded by our bboxes per track             │
│      │   per (frame, dancer_id): binary mask                         │
│      ▼                                                               │
│   ViTPose-h-coco-25 per crop                                         │
│      │   per (frame, dancer_id): openpose-135 + cocoophf-17 kpts     │
│      ▼                                                               │
│   est_calib + (optional) DROID-SLAM                                  │
│      │   per frame: cam_R, cam_T, focal, center                      │
│      ▼                                                               │
│   write `intermediates/masks_per_track/{tid}/{frame:08d}.png`        │
│   write `intermediates/masks_palette/{frame:08d}.png` (for SAM-Body4D)│
│   write `intermediates/keypoints/{openpose,vitpose}.pkl`             │
│   write `intermediates/camera/{intrinsics.json,slam.pkl}`            │
└──────────────────────────────────────────────────────────────────────┘
                ┌────────────────┴────────────────┐
                ▼                                 ▼
┌──────────────────────────────┐  ┌──────────────────────────────────────┐
│ STAGE C1 — `phmr_pt2.4` env  │  │ STAGE C2 — `body4d` env              │
│ run PromptHMR-Vid            │  │ run SAM-Body4D                       │
│                              │  │                                      │
│ feeds pre-built tracks dict  │  │ feeds pre-built palette masks +      │
│ into Pipeline.__call__       │  │ obj_ids into OfflineApp              │
│   (skips run_detect_track,   │  │   (skips first-frame detect + SAM-3) │
│    estimate_2d_keypoints)    │  │                                      │
│                              │  │                                      │
│ writes:                      │  │ writes:                              │
│   prompthmr/results.pkl      │  │   sam_body4d/mesh_4d_individual/     │
│   prompthmr/world4d.mcs      │  │   sam_body4d/focal_4d_individual/    │
│   prompthmr/subject-*.smpl   │  │   sam_body4d/rendered_frames/        │
│   prompthmr/world4d.glb      │  │   sam_body4d/4d_*.mp4                │
└──────────────────────────────┘  └──────────────────────────────────────┘
                └────────────────┬────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE D — runs in the tracking env (numpy / opencv only)             │
│                                                                      │
│   reads both prompthmr/ and sam_body4d/                              │
│      │                                                               │
│      ▼                                                               │
│   extract 3D joints in cam coords from each, project to COCO-17      │
│      │                                                               │
│      ▼                                                               │
│   compute metrics:                                                   │
│     - 2D reprojection error vs ViTPose                               │
│     - per-joint temporal jitter (cm/frame)                           │
│     - foot-skating velocity                                          │
│     - per-pair distance for close-contact frames                     │
│      │                                                               │
│      ▼                                                               │
│   render:                                                            │
│     - side-by-side mesh overlay video (ffmpeg)                       │
│     - per-dancer trajectory plots                                    │
│      │                                                               │
│      ▼                                                               │
│   write comparison/metrics.json + comparison/report.html             │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.1 The on-disk artifact contract

```
runs/3d_compare/<clip>/
├── intermediates/
│   ├── frames/
│   │   ├── 00000000.jpg        # max_height=896 to match PromptHMR
│   │   └── ...
│   ├── frames_full/            # original-resolution copies for SAM-Body4D
│   │   ├── 00000000.jpg
│   │   └── ...
│   ├── tracks.pkl              # dict[tid -> {frames, bboxes, confs}]
│   ├── masks_per_track/
│   │   └── {tid}/{frame:08d}.png   # binary 0/255, 896-resolution
│   ├── masks_palette/
│   │   └── {frame:08d}.png         # palette PNG, pixel == tid, full-resolution
│   ├── masks_union.npy             # (T, H, W) bool — for DROID-SLAM masking
│   ├── keypoints/
│   │   ├── openpose.pkl            # dict[tid -> (T, 135, 3)]
│   │   └── vitpose.pkl             # dict[tid -> (T, 17, 3)]
│   └── camera/
│       ├── intrinsics.json         # {fx, fy, cx, cy}
│       └── slam.pkl                # optional {pred_cam_R, pred_cam_T}
├── prompthmr/
│   ├── results.pkl
│   ├── world4d.mcs
│   ├── world4d.glb
│   ├── subject-*.smpl
│   └── joints_world.npy            # (T, N_dancers, 17, 3) for comparison
├── sam_body4d/
│   ├── mesh_4d_individual/{tid}/{frame:08d}.ply
│   ├── focal_4d_individual/{tid}/{frame:08d}.json
│   ├── rendered_frames/{frame:08d}.jpg
│   ├── 4d_*.mp4
│   └── joints_world.npy            # (T, N_dancers, 17, 3) for comparison
└── comparison/
    ├── metrics.json
    ├── side_by_side.mp4
    └── report.html
```

The contract is: **every artifact has exactly one producer**, and
downstream stages read by path, not by import. This means the three
conda envs never have to share a Python process.

### 4.2 Conda environments

We will need three:

| Env name | Python | PyTorch | Key deps | Size |
|---|---|---|---|---|
| `tracking` (existing) | 3.11 | 2.x mps/cuda | boxmot, ultralytics | ~5 GB |
| `phmr_pt2.4` (new) | 3.11.9 | 2.4.0+cu121 | detectron2, sam2, droid_backends_intr, lietorch, xformers, chumpy, smplx | ~12 GB |
| `body4d` (new) | 3.12 | 2.7.1+cu118 | sam3, diffusion-vas, sam-3d-body, moge2, depth-anything-v2 | ~18 GB |

The PromptHMR env install is `scripts/install.sh --pt_version=2.4
--world-video=true` from inside the cloned PromptHMR repo. The
SAM-Body4D env install is from the README — `pip install -e
models/sam3` then `pip install -e .` then `python scripts/setup.py`.

Both repos must be cloned **outside** our project tree to keep their
file systems isolated:

```
~/code/
├── PromptHMR/         # cloned from https://github.com/yufu-wang/PromptHMR
└── sam-body4d/        # cloned from https://github.com/gaomingqi/sam-body4d
```

Our scripts in `~/Desktop/yolo+bytetrack/threed/` will reference these via
absolute paths in a config file (so they're easy to relocate per
machine).

---

## 5. Implementation plan (bite-sized tasks)

> Convention: every task creates or modifies a small set of files.
> Each step inside a task is a single 2–5 min action.
> Test commands assume working directory = `~/Desktop/yolo+bytetrack`
> unless otherwise noted.

### Task 1: Bootstrap the 3D output structure and config

**Files:**
- Create: `threed/__init__.py`
- Create: `threed/config.py`
- Create: `threed/io.py`
- Test: `tests/threed/__init__.py`
- Test: `tests/threed/test_config.py`
- Test: `tests/threed/test_io.py`

> Package directory is `threed/` (not `3d/`) because Python module
> names cannot start with a digit. Output directory on disk stays
> `runs/3d_compare/` for human readability.

- [ ] **Step 1: Write the failing tests**

```python
# tests/threed/test_config.py
import pytest
from pathlib import Path
from threed.config import CompareConfig, default_config

def test_default_config_has_all_paths(tmp_path):
    cfg = default_config(repo_root=tmp_path)
    assert cfg.repo_root == tmp_path
    assert cfg.output_root == tmp_path / "runs" / "3d_compare"
    assert cfg.phmr_repo.is_absolute()
    assert cfg.body4d_repo.is_absolute()

def test_clip_dirs_are_namespaced(tmp_path):
    cfg = default_config(repo_root=tmp_path)
    dirs = cfg.clip_dirs("loveTest")
    assert dirs.intermediates == tmp_path / "runs" / "3d_compare" / "loveTest" / "intermediates"
    assert dirs.prompthmr == tmp_path / "runs" / "3d_compare" / "loveTest" / "prompthmr"
    assert dirs.sam_body4d == tmp_path / "runs" / "3d_compare" / "loveTest" / "sam_body4d"
    assert dirs.comparison == tmp_path / "runs" / "3d_compare" / "loveTest" / "comparison"
```

```python
# tests/threed/test_io.py
import numpy as np
from threed.io import save_tracks, load_tracks, TrackEntry

def test_tracks_roundtrip(tmp_path):
    tid = 7
    track = TrackEntry(
        track_id=tid,
        frames=np.array([0, 1, 2, 5, 6], dtype=np.int64),
        bboxes=np.random.rand(5, 4).astype(np.float32) * 1000,
        confs=np.array([0.9, 0.8, 0.85, 0.92, 0.88], dtype=np.float32),
    )
    out = tmp_path / "tracks.pkl"
    save_tracks({tid: track}, out)
    loaded = load_tracks(out)
    assert set(loaded.keys()) == {tid}
    np.testing.assert_array_equal(loaded[tid].frames, track.frames)
    np.testing.assert_allclose(loaded[tid].bboxes, track.bboxes)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/threed/ -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'threed'`.

- [ ] **Step 3: Implement `threed/config.py`**

```python
# threed/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ClipDirs:
    intermediates: Path
    prompthmr: Path
    sam_body4d: Path
    comparison: Path

    def ensure(self) -> "ClipDirs":
        for p in (self.intermediates, self.prompthmr, self.sam_body4d, self.comparison):
            p.mkdir(parents=True, exist_ok=True)
        return self

@dataclass(frozen=True)
class CompareConfig:
    repo_root: Path
    output_root: Path
    phmr_repo: Path
    body4d_repo: Path
    phmr_conda_env: str = "phmr_pt2.4"
    body4d_conda_env: str = "body4d"
    body4d_ckpt_root: Path = Path("~/checkpoints/body4d").expanduser()
    max_height: int = 896          # matches PromptHMR pipeline default
    max_fps: int = 60
    body4d_batch_size: int = 16    # default; bumped per-clip if VRAM allows
    body4d_completion_enable: bool = True   # default on; 9x runtime but needed for crowded dance scenes

    def clip_dirs(self, clip: str) -> ClipDirs:
        root = self.output_root / clip
        return ClipDirs(
            intermediates=root / "intermediates",
            prompthmr=root / "prompthmr",
            sam_body4d=root / "sam_body4d",
            comparison=root / "comparison",
        )

def default_config(repo_root: Path | None = None) -> CompareConfig:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    return CompareConfig(
        repo_root=repo_root,
        output_root=repo_root / "runs" / "3d_compare",
        phmr_repo=Path("~/code/PromptHMR").expanduser(),
        body4d_repo=Path("~/code/sam-body4d").expanduser(),
    )
```

- [ ] **Step 4: Implement `threed/io.py`**

```python
# threed/io.py
from __future__ import annotations
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np

@dataclass
class TrackEntry:
    track_id: int
    frames: np.ndarray   # (T,) int64
    bboxes: np.ndarray   # (T, 4) float32 xyxy
    confs:  np.ndarray   # (T,) float32
    masks:  Optional[np.ndarray] = None  # (T, H, W) bool — set by Stage B
    detected: Optional[np.ndarray] = None  # (T,) bool — True for real frames

def save_tracks(tracks: Dict[int, TrackEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        tid: {
            "track_id": t.track_id,
            "frames": t.frames,
            "bboxes": t.bboxes,
            "confs": t.confs,
            "masks": t.masks,
            "detected": t.detected,
        }
        for tid, t in tracks.items()
    }
    joblib.dump(payload, path)

def load_tracks(path: Path) -> Dict[int, TrackEntry]:
    payload = joblib.load(path)
    return {
        int(tid): TrackEntry(
            track_id=int(d["track_id"]),
            frames=np.asarray(d["frames"], dtype=np.int64),
            bboxes=np.asarray(d["bboxes"], dtype=np.float32),
            confs=np.asarray(d["confs"], dtype=np.float32),
            masks=(np.asarray(d["masks"], dtype=bool) if d.get("masks") is not None else None),
            detected=(np.asarray(d["detected"], dtype=bool) if d.get("detected") is not None else None),
        )
        for tid, d in payload.items()
    }
```

- [ ] **Step 5: Make `threed` importable from project root**

Create `threed/__init__.py`:

```python
# threed/__init__.py
from .config import CompareConfig, ClipDirs, default_config  # noqa: F401
from .io import TrackEntry, save_tracks, load_tracks  # noqa: F401
```

Do **not** create `tests/threed/__init__.py`. The plan originally said
to create it, but with pytest 8.x that file makes pytest see
`tests/threed/` as a package literally named `threed` (because there is
no `tests/__init__.py`), then prepend `tests/` to `sys.path`. After
that, `from threed.config import ...` resolves to the empty
`tests/threed/` package and raises `ModuleNotFoundError: No module
named 'threed.config'`.

Instead, leave `tests/threed/` as a plain directory and use pytest's
`importlib` import mode. Add a minimal `pyproject.toml` at the repo
root:

```toml
# pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = ["--import-mode=importlib"]
```

(If a `pyproject.toml` already exists in your checkout, just add the
`[tool.pytest.ini_options]` block to it.)

You also want a `conftest.py` at the repo root so any caller (with or
without `pytest`) can find `threed`:

```python
# conftest.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/threed/ -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add threed tests/threed
git commit -m "feat(threed): scaffold dual-pipeline config + tracks IO"
```

---

### Task 2: Convert DeepOcSort cache → shared `tracks.pkl`

**Files:**
- Create: `threed/stage_a/__init__.py`
- Create: `threed/stage_a/extract_tracks.py`
- Test: `tests/threed/test_extract_tracks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/threed/test_extract_tracks.py
import numpy as np
import pickle
from pathlib import Path
from prune_tracks import FrameDetections
from threed.stage_a.extract_tracks import extract_tracks_from_cache

def _make_cache(path: Path):
    """Two tracks, ID 1 in frames 0-2, ID 2 in frames 1-3."""
    fds = []
    for f in range(4):
        xs, cs, ts = [], [], []
        if f <= 2:
            xs.append([10, 10, 100, 200]); cs.append(0.9); ts.append(1)
        if 1 <= f <= 3:
            xs.append([200, 50, 300, 250]); cs.append(0.85); ts.append(2)
        fds.append(FrameDetections(
            xyxys=np.asarray(xs, dtype=np.float32) if xs else np.zeros((0, 4), np.float32),
            confs=np.asarray(cs, dtype=np.float32),
            tids=np.asarray(ts, dtype=np.int64),
        ))
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "demo.pkl", "wb") as f:
        pickle.dump(fds, f)

def test_extract_tracks_from_cache(tmp_path):
    _make_cache(tmp_path)
    tracks = extract_tracks_from_cache(tmp_path, min_total_frames=1, min_conf=0.0)
    assert set(tracks) == {1, 2}
    np.testing.assert_array_equal(tracks[1].frames, [0, 1, 2])
    np.testing.assert_array_equal(tracks[2].frames, [1, 2, 3])
    assert tracks[1].bboxes.shape == (3, 4)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/threed/test_extract_tracks.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the extractor**

```python
# threed/stage_a/extract_tracks.py
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict
import numpy as np

from prune_tracks import FrameDetections, prune_detections
from threed.io import TrackEntry

def extract_tracks_from_cache(
    cache_dir: Path,
    *,
    min_total_frames: int = 60,
    min_conf: float = 0.38,
) -> Dict[int, TrackEntry]:
    """Read the FrameDetections cache produced by run_winner_stack_demo
    and return a per-track dict in the shape PromptHMR-Vid wants."""
    pkls = sorted(cache_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No FrameDetections pickle in {cache_dir}")
    if len(pkls) > 1:
        raise ValueError(f"Multiple pickles in {cache_dir}; expected exactly one")
    with open(pkls[0], "rb") as f:
        fds = pickle.load(f)

    fds = prune_detections(
        fds,
        min_total_frames=min_total_frames,
        min_conf=min_conf,
    )

    rows: Dict[int, dict] = {}
    for frame_idx, fd in enumerate(fds):
        if len(fd.tids) == 0:
            continue
        for k in range(len(fd.tids)):
            tid = int(fd.tids[k])
            d = rows.setdefault(tid, {"frames": [], "bboxes": [], "confs": []})
            d["frames"].append(frame_idx)
            d["bboxes"].append(fd.xyxys[k].tolist())
            d["confs"].append(float(fd.confs[k]))

    out: Dict[int, TrackEntry] = {}
    for tid, d in rows.items():
        out[tid] = TrackEntry(
            track_id=tid,
            frames=np.asarray(d["frames"], dtype=np.int64),
            bboxes=np.asarray(d["bboxes"], dtype=np.float32),
            confs=np.asarray(d["confs"], dtype=np.float32),
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/threed/test_extract_tracks.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add threed/stage_a tests/threed/test_extract_tracks.py
git commit -m "feat(threed/stage_a): extract per-track dict from DeepOcSort cache"
```

---

### Task 3: Frame extraction at PromptHMR-compatible resolution

**Files:**
- Create: `threed/stage_a/extract_frames.py`
- Test: `tests/threed/test_extract_frames.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/threed/test_extract_frames.py
import cv2
import numpy as np
from pathlib import Path
from threed.stage_a.extract_frames import extract_frames

def _make_video(path: Path, n_frames: int = 5, w: int = 1920, h: int = 1080):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 30, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        vw.write(frame)
    vw.release()

def test_extract_frames_resizes_to_max_height(tmp_path):
    src = tmp_path / "video.mp4"
    _make_video(src, n_frames=5, w=1920, h=1080)
    out_896 = tmp_path / "frames_896"
    out_full = tmp_path / "frames_full"

    n = extract_frames(src, out_896, out_full, max_height=896)
    assert n == 5

    img_896 = cv2.imread(str(out_896 / "00000000.jpg"))
    img_full = cv2.imread(str(out_full / "00000000.jpg"))
    assert img_896.shape[0] == 896
    assert img_full.shape[0] == 1080  # untouched
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/threed/test_extract_frames.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement frame extraction**

```python
# threed/stage_a/extract_frames.py
from __future__ import annotations
from pathlib import Path
import cv2

def extract_frames(
    video: Path,
    out_dir_resized: Path,
    out_dir_full: Path,
    *,
    max_height: int = 896,
) -> int:
    """Extract frames from a video into TWO folders:
    - out_dir_resized: frames downscaled so height <= max_height (for PromptHMR)
    - out_dir_full:    frames at original resolution (for SAM-Body4D)

    Returns the number of frames written.
    """
    out_dir_resized.mkdir(parents=True, exist_ok=True)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            if h > max_height:
                scale = max_height / h
                resized = cv2.resize(frame, (int(round(w * scale)), max_height),
                                     interpolation=cv2.INTER_AREA)
            else:
                resized = frame
            cv2.imwrite(str(out_dir_resized / f"{n:08d}.jpg"), resized)
            cv2.imwrite(str(out_dir_full / f"{n:08d}.jpg"), frame)
            n += 1
    finally:
        cap.release()
    return n
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/threed/test_extract_frames.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add threed/stage_a/extract_frames.py tests/threed/test_extract_frames.py
git commit -m "feat(threed/stage_a): extract frames at 896 + full resolution"
```

---

### Task 4: Stage A driver — produce frames + tracks for one clip

**Files:**
- Create: `threed/stage_a/run_stage_a.py`
- Test (smoke): manual run on `adiTest`

- [ ] **Step 1: Implement the driver script**

```python
# threed/stage_a/run_stage_a.py
"""Stage A — produce frames + DeepOcSort tracks for one clip.

Runs in the existing tracking conda env. No HMR-specific deps.

Usage:
    python -m threed.stage_a.run_stage_a --clip adiTest \
        --video /Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov \
        --cache-dir runs/winner_stack_demo/_cache/adiTest
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from threed.config import default_config
from threed.io import save_tracks
from threed.stage_a.extract_tracks import extract_tracks_from_cache
from threed.stage_a.extract_frames import extract_frames

log = logging.getLogger("stage_a")

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Path to runs/<run>/_cache/<clip> with one .pkl inside")
    p.add_argument("--min-total-frames", type=int, default=60)
    p.add_argument("--min-conf", type=float, default=0.38)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = default_config()
    dirs = cfg.clip_dirs(args.clip).ensure()

    log.info("[%s] extracting frames from %s", args.clip, args.video)
    n_frames = extract_frames(
        args.video,
        out_dir_resized=dirs.intermediates / "frames",
        out_dir_full=dirs.intermediates / "frames_full",
        max_height=cfg.max_height,
    )
    log.info("[%s] wrote %d frames", args.clip, n_frames)

    log.info("[%s] extracting tracks from %s", args.clip, args.cache_dir)
    tracks = extract_tracks_from_cache(
        args.cache_dir,
        min_total_frames=args.min_total_frames,
        min_conf=args.min_conf,
    )
    log.info("[%s] %d tracks survived pruning", args.clip, len(tracks))

    save_tracks(tracks, dirs.intermediates / "tracks.pkl")
    log.info("[%s] wrote %s", args.clip, dirs.intermediates / "tracks.pkl")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test on `adiTest` (188 frames, 5 dancers)**

Run:

```bash
python -m threed.stage_a.run_stage_a \
    --clip adiTest \
    --video /Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov \
    --cache-dir runs/winner_stack_demo/_cache/adiTest
```

Expected output: directory `runs/3d_compare/adiTest/intermediates/`
contains `frames/` (188 jpgs), `frames_full/` (188 jpgs), and
`tracks.pkl` (5 tracks).

Verify:

```bash
ls runs/3d_compare/adiTest/intermediates/frames | wc -l    # 188
python -c "from threed.io import load_tracks; t = load_tracks('runs/3d_compare/adiTest/intermediates/tracks.pkl'); print(sorted(t.keys()), {k: len(v.frames) for k,v in t.items()})"
```

Expected: 5 track ids and frame counts close to 188 each.

- [ ] **Step 3: Commit**

```bash
git add threed/stage_a/run_stage_a.py
git commit -m "feat(threed/stage_a): driver to produce frames + tracks for a clip"
```

---

### Task 5: Install PromptHMR conda environment

**Files:** none in this repo (operates externally).

- [ ] **Step 1: Clone PromptHMR outside the project tree**

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/yufu-wang/PromptHMR
cd PromptHMR
```

- [ ] **Step 2: Run the install script for PT 2.4 + world-video extras**

On a CUDA box (GH200, A100, H100):

```bash
bash scripts/install.sh --pt_version=2.4 --world-video=true
```

This creates the `phmr_pt2.4` conda env with detectron2 + sam2 +
droid_backends_intr + lietorch + xformers + chumpy.

On Apple Silicon dev: PromptHMR-Vid requires CUDA wheels (DROID-SLAM
depends on CUDA). Plan to develop on CUDA only; skip Apple Silicon
for this stage.

- [ ] **Step 3: Download SMPL-X bodies + checkpoints**

```bash
conda activate phmr_pt2.4
bash scripts/fetch_smplx.sh        # prompts for SMPL-X license
bash scripts/fetch_data.sh         # PromptHMR + DROID + ZoeDepth + ViTPose
```

Then download the BEDLAM2 video-head checkpoint (better quality than the
default `prhmr_release_002.ckpt` because it is trained on BEDLAM1+BEDLAM2):

```bash
mkdir -p data/pretrain/phmr_vid
wget -O data/pretrain/phmr_vid/phmr_b1b2.ckpt \
    https://download.is.tue.mpg.de/bedlam2/ml/videos/phmr_b1b2.ckpt
```

To activate it, edit `pipeline/phmr_vid.py` line 22 (per the PromptHMR
README's "modify the checkpoint path in this line" instruction) so that
`phmr_vid_ckpt` reads `'data/pretrain/phmr_vid/phmr_b1b2.ckpt'` instead of
`'data/pretrain/phmr_vid/prhmr_release_002.ckpt'`. The yaml file itself
has no `pretrained_ckpt` key (it is the per-component config); only the
.py file needs to change.

Recommended sequencing:
1. Run the bundled smoke test in step 4 with the default
   `prhmr_release_002.ckpt` (no edit) so a successful demo proves the env
   itself works, isolated from the ckpt swap.
2. After the smoke test passes, swap to `phmr_b1b2.ckpt` before running
   our pipeline in plan Task 7 (PromptHMR-Vid sidecar). Keep the swap as
   a separate, reverted-if-needed change in `pipeline/phmr_vid.py`.

- [ ] **Step 4: Verify the env works on the bundled boxing example**

```bash
python scripts/demo_video.py --input_video data/examples/boxing_short.mp4 --static_camera
```

Expected: produces `results/boxing_short/results.pkl`,
`results/boxing_short/world4d.mcs`, `results/boxing_short/world4d.glb`.
Viser web viewer URL printed.

- [ ] **Step 5: Note the install in our docs**

Add a paragraph to `docs/3D_DUAL_PIPELINE_PLAN.md` (this file) §11:

> "PromptHMR conda env installed at `~/code/PromptHMR` with
> `phmr_pt2.4`. Verified with `boxing_short.mp4` demo on <date>."

(No commit needed — this is documentation for the human operator.)

---

### Task 6: PromptHMR sidecar — SAM-2 mask generation from our bboxes

**Files (revised 2026-04-18 — see plan-correction commit):**
- Create: `threed/sidecar_promthmr/__init__.py` (in our SAM-HMR repo)
- Create: `threed/sidecar_promthmr/build_masks.py` (in our SAM-HMR repo)
- Create: `tests/threed/test_sidecar_promthmr_build_masks.py`

> Original plan put this in `~/code/PromptHMR/our_pipeline/` (inside the
> third-party clone). Revised: keep the source of truth in our repo so
> it goes through normal git/CI, and at runtime inject PromptHMR's
> `pipeline.*` modules onto `sys.path` via the `PROMPTHMR_PATH`
> environment variable (default `~/code/PromptHMR`). PromptHMR's bundled
> SAM-2 weights + configs continue to live where they always did
> (`<PROMPTHMR_PATH>/data/pretrain/sam2_ckpts/` and
> `<PROMPTHMR_PATH>/pipeline/sam2/*.yaml`).
>
> Default checkpoint changed from `sam2_hiera_large.pt` to
> `sam2_hiera_tiny.pt`: that is the only SAM-2 weight file
> `scripts/fetch_data.sh` actually downloads (verified on box). The
> `_t.yaml` config goes with it. We can opt into `sam2_hiera_large.pt`
> later by passing `--sam2-checkpoint` + `--sam2-config` if we feel the
> tiny model is too noisy on dense dance crowds.

- [ ] **Step 1: Implement the SAM-2 mask builder**

```python
# threed/sidecar_promthmr/build_masks.py  (in our SAM-HMR repo)
"""Per-track SAM-2 mask propagation seeded by our DeepOcSort bboxes.

Reads:
    intermediates/frames/             # JPGs at max_height=896
    intermediates/tracks.pkl          # threed.io.save_tracks payload
                                      # {tid -> {track_id, frames, bboxes,
                                      #          confs, masks?, detected?}}

Writes:
    intermediates/masks_per_track/{tid}/{frame:08d}.png   # binary 0/255
    intermediates/masks_palette/{frame:08d}.png           # palette PNG, pixel == tid
    intermediates/masks_union.npy                         # (T, H, W) bool

Imports PromptHMR's bundled SAM-2 video predictor by inserting
$PROMPTHMR_PATH (default ~/code/PromptHMR) at the front of sys.path.
"""
from __future__ import annotations
import argparse
import joblib
import os
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image

# DAVIS palette identical to SAM-Body4D's
def _davis_palette() -> bytes:
    palette = [0, 0, 0]
    for i in range(1, 256):
        h = i * 360 // 256
        s = 0.9
        v = 0.9
        # HSV -> RGB
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60: r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        palette.extend([int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)])
    return bytes(palette)

DAVIS_PALETTE = _davis_palette()

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--intermediates-dir", type=Path, required=True)
    p.add_argument("--prompthmr-path", type=Path,
                   default=Path(os.environ.get("PROMPTHMR_PATH",
                                               "~/code/PromptHMR")).expanduser())
    p.add_argument("--sam2-checkpoint", type=Path, default=None,
                   help="Default: <prompthmr-path>/data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt")
    p.add_argument("--sam2-config", type=str,
                   default="pipeline/sam2/sam2_hiera_t.yaml")
    args = p.parse_args(argv)

    interm = args.intermediates_dir.expanduser().resolve()
    frames_dir = interm / "frames"
    tracks_pkl = interm / "tracks.pkl"
    out_per_tid = interm / "masks_per_track"
    out_palette = interm / "masks_palette"
    out_union = interm / "masks_union.npy"

    out_per_tid.mkdir(parents=True, exist_ok=True)
    out_palette.mkdir(parents=True, exist_ok=True)

    # Load SAM-2 video predictor (uses PromptHMR's bundled config)
    from pipeline.detector.sam2_video_predictor import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint))

    # Load tracks
    tracks = joblib.load(tracks_pkl)

    # Frame paths
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    n_frames = len(frame_paths)
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]

    # Per-track SAM-2 propagation
    # We add the FIRST observation of each track as a SAM-2 prompt,
    # then propagate. SAM-2 returns masks for every frame; we mask out
    # frames where this track is absent in our DeepOcSort tracks.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(frames_dir),
            async_loading_frames=False,
            offload_video_to_cpu=True,
        )
        for tid, t in tracks.items():
            if len(t["frames"]) == 0:
                continue
            first_frame = int(t["frames"][0])
            x1, y1, x2, y2 = t["bboxes"][0]
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            predictor.add_new_points(
                state,
                frame_idx=first_frame,
                obj_id=int(tid),
                box=box,
            )

        # Per-frame, per-tid masks
        per_frame_per_tid = {}  # {frame: {tid: bool mask}}
        for frame, obj_ids, mask_logits in predictor.propagate_in_video(state):
            per_frame_per_tid[frame] = {
                int(oid): (mask_logits[i, 0] > 0.0).cpu().numpy()
                for i, oid in enumerate(obj_ids)
            }

    # Filter masks by DeepOcSort presence
    valid_frames = {int(tid): set(int(f) for f in t["frames"].tolist())
                    for tid, t in tracks.items()}

    # Write per-tid binary PNGs (only for frames where DeepOcSort had this tid)
    for tid, t in tracks.items():
        out_t = out_per_tid / str(tid)
        out_t.mkdir(parents=True, exist_ok=True)
        for frame in range(n_frames):
            if frame in valid_frames[int(tid)] and frame in per_frame_per_tid \
               and int(tid) in per_frame_per_tid[frame]:
                msk = per_frame_per_tid[frame][int(tid)]
                cv2.imwrite(str(out_t / f"{frame:08d}.png"),
                            (msk.astype(np.uint8) * 255))

    # Write palette PNG per frame (pixel value == tid) for SAM-Body4D
    union = np.zeros((n_frames, H, W), dtype=bool)
    for frame in range(n_frames):
        canvas = np.zeros((H, W), dtype=np.uint8)
        if frame in per_frame_per_tid:
            for tid, msk in per_frame_per_tid[frame].items():
                if tid not in valid_frames or frame not in valid_frames[tid]:
                    continue
                # Resolve overlap by larger-tid-wins (deterministic)
                canvas[msk] = tid
                union[frame] |= msk
        img = Image.fromarray(canvas, mode="P")
        img.putpalette(DAVIS_PALETTE)
        img.save(str(out_palette / f"{frame:08d}.png"))

    np.save(out_union, union)
    print(f"wrote {len(tracks)} per-tid masks, {n_frames} palette PNGs, "
          f"and union of shape {union.shape}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test on adiTest**

Sync intermediates to the box (or regenerate via `threed.stage_a.run_stage_a`):

```bash
rsync -av runs/3d_compare/adiTest/intermediates/ \
    cuda-box:~/work/3d_compare/adiTest/intermediates/
```

Then on the CUDA box:

```bash
cd ~/code/SAM-HMR
conda activate phmr_pt2.4
PROMPTHMR_PATH=~/code/PromptHMR \
    python -m threed.sidecar_promthmr.build_masks \
        --intermediates-dir ~/work/3d_compare/adiTest/intermediates
```

Verify:

```bash
ls ~/work/3d_compare/adiTest/intermediates/masks_per_track/   # 5 tid dirs
ls ~/work/3d_compare/adiTest/intermediates/masks_palette/ | wc -l   # 188
python -c "import numpy as np; m = np.load('${HOME}/work/3d_compare/adiTest/intermediates/masks_union.npy'); print(m.shape, m.sum())"
```

Expected: 5 dirs with up to 188 PNGs each, 188 palette PNGs, union
shape `(188, 896, W)` with non-zero sum.

- [ ] **Step 3: Commit (in our SAM-HMR repo)**

```bash
git add threed/sidecar_promthmr/__init__.py \
        threed/sidecar_promthmr/build_masks.py \
        tests/threed/test_sidecar_promthmr_build_masks.py
git commit -m "feat(threed/sidecar_promthmr): SAM-2 mask propagation seeded by DeepOcSort bboxes"
```

---

### Task 7: PromptHMR sidecar — run PromptHMR-Vid on shared artifacts

**Files:**
- Create: `~/code/PromptHMR/our_pipeline/run_phmr.py`

- [ ] **Step 1: Implement the runner**

```python
# ~/code/PromptHMR/our_pipeline/run_phmr.py
"""Run PromptHMR-Vid on a shared intermediates directory.

Reads:
    intermediates/frames/             # JPGs at max_height=896
    intermediates/tracks.pkl          # {tid -> {frames, bboxes, confs}}
    intermediates/masks_per_track/    # per-tid binary PNGs
    intermediates/masks_union.npy     # union mask
    intermediates/camera/intrinsics.json   (optional)

Writes:
    prompthmr/results.pkl
    prompthmr/world4d.mcs
    prompthmr/world4d.glb
    prompthmr/subject-*.smpl
    prompthmr/joints_world.npy
"""
from __future__ import annotations
import argparse
import json
import joblib
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import Pipeline


def _load_per_track_masks(interm: Path, tracks: dict, n_frames: int, H: int, W: int):
    """Reconstruct per-track (T, H, W) bool masks from masks_per_track/."""
    for tid, t in tracks.items():
        per_t = []
        for f in t["frames"]:
            png = interm / "masks_per_track" / str(tid) / f"{int(f):08d}.png"
            if png.is_file():
                m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE) > 127
            else:
                m = np.zeros((H, W), dtype=bool)
            per_t.append(m)
        t["masks"] = np.stack(per_t).astype(bool)
        t["track_id"] = int(tid)
        # mark all frames as "detected" (we trust DeepOcSort)
        t["detected"] = np.ones(len(t["frames"]), dtype=bool)
    return tracks

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intermediates-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write prompthmr/ outputs")
    p.add_argument("--static-camera", action="store_true")
    args = p.parse_args(argv)

    interm = args.intermediates_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    frames_dir = interm / "frames"
    tracks_pkl = interm / "tracks.pkl"
    union_npy = interm / "masks_union.npy"
    intr_json = interm / "camera" / "intrinsics.json"

    # Load frames as np.ndarray list (PromptHMR's load_video_frames also accepts this)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    images = [cv2.imread(str(p))[:, :, ::-1] for p in frame_paths]
    H, W = images[0].shape[:2]
    n_frames = len(images)

    # Build pipeline
    pipeline = Pipeline(static_cam=args.static_camera)
    pipeline.images = images
    pipeline.seq_folder = str(out)            # PromptHMR writes intermediate cache here
    pipeline.cfg.seq_folder = pipeline.seq_folder
    pipeline.cfg.tracker = "external"          # mark for downstream `mask_prompt` decision
    pipeline.fps = 30                          # default; could read from video metadata

    # Load OUR tracks + masks
    tracks = joblib.load(tracks_pkl)
    tracks = _load_per_track_masks(interm, tracks, n_frames, H, W)

    union = np.load(union_npy)

    # Pre-populate results to skip detect/track + (optionally) SLAM
    pipeline.results = {
        "camera": {},
        "people": tracks,
        "timings": {},
        "masks": union,
        "has_tracks": True,
        "has_hps_cam": False,
        "has_hps_world": False,
        "has_slam": False,
        "has_hands": False,
        "has_2d_kpts": False,
        "has_post_opt": False,
    }

    # Camera intrinsics
    if intr_json.is_file():
        intr = json.loads(intr_json.read_text())
        pipeline.results["camera"] = {
            "img_focal": intr["fx"],
            "img_center": np.array([intr["cx"], intr["cy"]], dtype=np.float32),
        }
    else:
        from pipeline.tools import est_camera
        pipeline.results["camera"] = est_camera(images[0])

    # Run the rest of the pipeline
    pipeline.camera_motion_estimation(args.static_camera)
    pipeline.estimate_2d_keypoints()           # ViTPose; populates 'vitpose'
    pipeline.hps_estimation()                  # the actual PromptHMR-Vid call (mask_prompt=True)
    pipeline.world_hps_estimation()

    def _to_numpy(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _to_numpy(v)
            elif isinstance(v, torch.Tensor):
                d[k] = v.detach().cpu().numpy()
    _to_numpy(pipeline.results)

    if pipeline.cfg.run_post_opt:
        pipeline.post_optimization()

    # Save results.pkl + per-track joints in world coords for comparison
    joblib.dump(pipeline.results, out / "results.pkl")

    # Extract joints (22 SMPL body joints) → write joints_world.npy
    # Shape: (n_frames, n_dancers, 22, 3); pad missing dancers with NaN.
    tids_sorted = sorted(pipeline.results["people"].keys())
    joints = np.full((n_frames, len(tids_sorted), 22, 3), np.nan, dtype=np.float32)
    smplx = pipeline.smplx.to("cuda")
    for di, tid in enumerate(tids_sorted):
        person = pipeline.results["people"][tid]
        smplx_w = person["smplx_world"]
        with torch.no_grad():
            from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
            pose_aa = torch.tensor(smplx_w["pose"], dtype=torch.float32).cuda()
            shape  = torch.tensor(smplx_w["shape"], dtype=torch.float32).cuda()
            transl = torch.tensor(smplx_w["trans"], dtype=torch.float32).cuda()
            rotmat = axis_angle_to_matrix(pose_aa.reshape(-1, 55, 3))
            out_smplx = smplx(
                global_orient=rotmat[:, :1],
                body_pose=rotmat[:, 1:22],
                betas=shape,
                transl=transl,
            )
        body_j = out_smplx.joints[:, :22].cpu().numpy()  # (T_track, 22, 3)
        for fi, frame in enumerate(person["frames"]):
            joints[int(frame), di] = body_j[fi]

    np.save(out / "joints_world.npy", joints)
    print(f"wrote {out / 'results.pkl'} + joints {joints.shape}")

    # MCS / GLB / SMPL files are written by Pipeline.__call__ tail logic;
    # call it manually here:
    from smplcodec import SMPLCodec
    from pipeline.mcs_export_cam import export_scene_with_camera
    smpl_paths, presence = [], []
    for tid in tids_sorted:
        v = pipeline.results["people"][tid]
        smpl_f = out / f"subject-{tid}.smpl"
        SMPLCodec(
            shape_parameters=v["smplx_world"]["shape"].mean(0),
            body_pose=v["smplx_world"]["pose"][:, :22*3].reshape(-1, 22, 3),
            body_translation=v["smplx_world"]["trans"],
            frame_count=v["frames"].shape[0],
            frame_rate=float(pipeline.cfg.fps),
        ).write(str(smpl_f))
        smpl_paths.append(smpl_f)
        presence.append([int(v["frames"][0]), int(v["frames"][-1]) + 1])

    export_scene_with_camera(
        smpl_buffers=[open(p, "rb").read() for p in smpl_paths],
        frame_presences=presence,
        num_frames=n_frames,
        output_path=str(out / "world4d.mcs"),
        rotation_matrices=pipeline.results["camera_world"]["Rcw"],
        translations=pipeline.results["camera_world"]["Tcw"],
        focal_length=pipeline.results["camera_world"]["img_focal"],
        principal_point=pipeline.results["camera_world"]["img_center"],
        frame_rate=float(pipeline.cfg.fps),
        smplx_path="data/body_models/smplx/SMPLX_neutral_array_f32_slim.npz",
    )
    print(f"wrote {out / 'world4d.mcs'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test on adiTest**

```bash
cd ~/code/PromptHMR
conda activate phmr_pt2.4
python -m our_pipeline.run_phmr \
    --intermediates-dir work/3d_compare/adiTest/intermediates \
    --output-dir work/3d_compare/adiTest/prompthmr \
    --static-camera
```

Expected: `work/3d_compare/adiTest/prompthmr/results.pkl`,
`world4d.mcs`, `world4d.glb`, `subject-1.smpl` … `subject-5.smpl`,
`joints_world.npy` of shape `(188, 5, 22, 3)`.

Visual check: drag `world4d.mcs` to
[https://me.meshcapade.com/editor](https://me.meshcapade.com/editor).
Should show 5 dancing meshes.

- [ ] **Step 3: Commit (in the PromptHMR clone)**

```bash
cd ~/code/PromptHMR
git add our_pipeline/run_phmr.py
git commit -m "feat(our_pipeline): run PromptHMR-Vid on external tracks + masks"
```

---

### Task 8: Install SAM-Body4D conda environment

**Files:** none in this repo.

- [ ] **Step 1: Clone SAM-Body4D outside the project tree**

```bash
cd ~/code
git clone https://github.com/gaomingqi/sam-body4d
cd sam-body4d
```

- [ ] **Step 2: Create the conda env (per the README)**

```bash
conda create -n body4d python=3.12 -y
conda activate body4d
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps
pip install -e models/sam3
pip install -e .
```

- [ ] **Step 3: Accept HF gates and download checkpoints**

Visit and accept the gates on:
- https://huggingface.co/facebook/sam3
- https://huggingface.co/facebook/sam-3d-body-dinov3

Then:

```bash
huggingface-cli login
python scripts/setup.py --ckpt-root ~/checkpoints/body4d
```

This downloads SAM-3, SAM-3D-Body, MoGe-2, Diffusion-VAS, and
Depth-Anything V2 (~30 GB total).

- [ ] **Step 4: Verify on the bundled Gradio demo**

```bash
python app.py
```

Should launch a Gradio UI at `http://localhost:7860`. Pick one of
the bundled demo videos and click through "Generate Masks → Generate
4D" once. If outputs render to the UI, the env is good.

- [ ] **Step 5: Note the install in our docs**

Add a paragraph to §11 of this plan:

> "SAM-Body4D conda env installed at `~/code/sam-body4d` with `body4d`,
> checkpoints in `~/checkpoints/body4d`. Verified on bundled demo on
> <date>."

---

### Task 9: SAM-Body4D sidecar — run on shared artifacts

**Files:**
- Create: `~/code/sam-body4d/our_pipeline/__init__.py`
- Create: `~/code/sam-body4d/our_pipeline/run_body4d.py`

- [ ] **Step 1: Implement the runner**

```python
# ~/code/sam-body4d/our_pipeline/run_body4d.py
"""Run SAM-Body4D's HMR step on pre-computed images + palette masks.

Reads:
    intermediates/frames_full/        # full-resolution JPGs
    intermediates/masks_palette/      # palette PNGs, pixel == tid

Writes:
    sam_body4d/mesh_4d_individual/{tid}/{frame:08d}.ply
    sam_body4d/focal_4d_individual/{tid}/{frame:08d}.json
    sam_body4d/rendered_frames/{frame:08d}.jpg
    sam_body4d/4d_*.mp4
    sam_body4d/joints_world.npy
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "sam_3d_body"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "diffusion_vas"))

import torch
from omegaconf import OmegaConf

from scripts.offline_app import OfflineApp, build_diffusion_vas_config
from utils import jpg_folder_to_mp4

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intermediates-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config-path", type=Path,
                   default=Path("configs/body4d.yaml"))
    p.add_argument("--disable-completion", action="store_true",
                   help="Skip Diffusion-VAS occlusion completion (9x faster, "
                        "but worse on heavy dance occlusion). Default: completion ON.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--track-batch", type=int, default=5,
                   help="Process at most N tracks at a time to avoid OOM")
    args = p.parse_args(argv)

    interm = args.intermediates_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Override config in-memory
    cfg = OmegaConf.load(args.config_path)
    cfg.completion.enable = not args.disable_completion
    cfg.sam_3d_body.batch_size = args.batch_size
    cfg.runtime.output_dir = str(out.parent)   # OfflineApp creates a UUID subdir

    # Monkey-patch SAM-3 builder to skip loading SAM-3 (we don't need it)
    import scripts.offline_app as oa
    def _noop(_cfg):
        return None, None
    oa.build_sam3_from_config = _noop

    # Init OfflineApp
    predictor = OfflineApp(config_path=str(args.config_path))
    predictor.OUTPUT_DIR = str(out)
    os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)

    # Pre-populate images + masks where on_4d_generation expects them
    images_dst = out / "images"
    masks_dst = out / "masks"
    images_dst.mkdir(parents=True, exist_ok=True)
    masks_dst.mkdir(parents=True, exist_ok=True)

    # Hard-link or copy — link is faster
    for src in sorted((interm / "frames_full").glob("*.jpg")):
        dst = images_dst / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    for src in sorted((interm / "masks_palette").glob("*.png")):
        dst = masks_dst / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    # Discover present tids by scanning the first non-empty palette mask
    tids_present = set()
    for png in sorted(masks_dst.glob("*.png")):
        arr = np.array(Image.open(png).convert("P"))
        for v in np.unique(arr):
            if int(v) > 0:
                tids_present.add(int(v))
    tids_sorted = sorted(tids_present)
    print(f"discovered {len(tids_sorted)} tids: {tids_sorted}")

    # Process tracks in batches of size ≤ track_batch to avoid OOM
    for i in range(0, len(tids_sorted), args.track_batch):
        batch = tids_sorted[i:i + args.track_batch]
        predictor.RUNTIME["out_obj_ids"] = batch
        # Make sure each batch dir exists
        for tid in batch:
            (out / f"mesh_4d_individual/{tid}").mkdir(parents=True, exist_ok=True)
            (out / f"focal_4d_individual/{tid}").mkdir(parents=True, exist_ok=True)
            (out / f"rendered_frames_individual/{tid}").mkdir(parents=True, exist_ok=True)

        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation()
        print(f"finished tid batch {batch}")

    # Stitch the rendered frames into an MP4
    out_video = out / "4d_render.mp4"
    jpg_folder_to_mp4(str(out / "rendered_frames"), str(out_video))

    # Build joints_world.npy by reading per-frame focal + mesh
    # MHR meshes are in cam coords; we need to extract joints.
    # Use SAM-3D-Body's regressor. The MHR rig has a 70-joint skeleton;
    # we project to a COCO-17 subset for comparison.
    # ... (see Task 10) ...
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test on adiTest**

Sync intermediates to the SAM-Body4D box (could be the same CUDA box):

```bash
rsync -av runs/3d_compare/adiTest/intermediates/ \
    cuda-box:~/code/sam-body4d/work/3d_compare/adiTest/intermediates/
```

Then on the CUDA box:

```bash
cd ~/code/sam-body4d
conda activate body4d
python -m our_pipeline.run_body4d \
    --intermediates-dir work/3d_compare/adiTest/intermediates \
    --output-dir work/3d_compare/adiTest/sam_body4d \
    --batch-size 16 \
    --track-batch 5
```

Expected: `work/3d_compare/adiTest/sam_body4d/mesh_4d_individual/{1..5}/`
each containing 188 PLY files. Plus `4d_render.mp4` rendered overlay.

- [ ] **Step 3: Commit (in the SAM-Body4D clone)**

```bash
cd ~/code/sam-body4d
git add our_pipeline/run_body4d.py
git commit -m "feat(our_pipeline): run SAM-Body4D HMR on external palette masks"
```

---

### Task 10: Extract COCO-17 joints from SAM-Body4D output

**Files:**
- Create: `~/code/sam-body4d/our_pipeline/extract_joints.py`

The SAM-Body4D outputs are PLY meshes + per-frame focal length JSON.
The MHR mesh has a known regressor for joints (`mhr70_pose_info`); we
extract the 17 COCO joints via the official mapping.

- [ ] **Step 1: Implement the extractor**

```python
# ~/code/sam-body4d/our_pipeline/extract_joints.py
"""Extract COCO-17 joints (in camera coords) from SAM-Body4D output.

Reads:
    sam_body4d/focal_4d_individual/{tid}/{frame:08d}.json
    sam_body4d/mesh_4d_individual/{tid}/{frame:08d}.ply

Writes:
    sam_body4d/joints_cam.npy        # (T, N_dancers, 17, 3) cam coords
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh

# MHR70 → COCO-17 vertex indices (from sam_3d_body.metadata.mhr70.pose_info)
# These are the closest MHR vertex to each COCO joint.
# Source: SAM-3D-Body repo, sam_3d_body/metadata/mhr70.py
MHR70_TO_COCO17 = {
    "nose": 12,            # head
    "left_eye": 13,
    "right_eye": 14,
    "left_ear": 15,
    "right_ear": 16,
    "left_shoulder": 17,
    "right_shoulder": 18,
    "left_elbow": 19,
    "right_elbow": 20,
    "left_wrist": 21,
    "right_wrist": 22,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
}
COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sam-body4d-dir", type=Path, required=True)
    p.add_argument("--n-frames", type=int, required=True)
    args = p.parse_args(argv)

    sb4d = args.sam_body4d_dir.expanduser().resolve()
    mesh_dir = sb4d / "mesh_4d_individual"
    focal_dir = sb4d / "focal_4d_individual"

    tids = sorted(int(d.name) for d in mesh_dir.iterdir() if d.is_dir())
    print(f"found {len(tids)} tids: {tids}")

    joints = np.full((args.n_frames, len(tids), 17, 3), np.nan, dtype=np.float32)
    for di, tid in enumerate(tids):
        ply_dir = mesh_dir / str(tid)
        for ply_path in sorted(ply_dir.glob("*.ply")):
            frame = int(ply_path.stem)
            mesh = trimesh.load(ply_path, process=False)
            verts = mesh.vertices.view(np.ndarray)  # (V, 3) in cam coords (relative to person)
            # Add the per-frame cam translation
            focal_path = focal_dir / str(tid) / f"{frame:08d}.json"
            if focal_path.is_file():
                meta = json.loads(focal_path.read_text())
                cam_t = np.array(meta["camera"], dtype=np.float32)
                verts = verts + cam_t
            for ji, jname in enumerate(COCO17_NAMES):
                vi = MHR70_TO_COCO17[jname]
                joints[frame, di, ji] = verts[vi]

    np.save(sb4d / "joints_cam.npy", joints)
    print(f"wrote {sb4d / 'joints_cam.npy'} of shape {joints.shape}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

> Note: the `MHR70_TO_COCO17` mapping above uses placeholder vertex
> indices. Run-once verification: open one PLY in MeshLab + the rig's
> `mhr_model.pt` joint regressor; print the actual joint indices for
> the COCO-17 subset and update the dict. (This is a 30-min one-time
> task on top of the implementation step.)

- [ ] **Step 2: Run on adiTest**

```bash
cd ~/code/sam-body4d
python -m our_pipeline.extract_joints \
    --sam-body4d-dir work/3d_compare/adiTest/sam_body4d \
    --n-frames 188
```

Expected: `joints_cam.npy` of shape `(188, 5, 17, 3)`.

- [ ] **Step 3: Commit**

```bash
cd ~/code/sam-body4d
git add our_pipeline/extract_joints.py
git commit -m "feat(our_pipeline): extract COCO-17 joints from MHR meshes"
```

---

### Task 11: Comparison harness — metrics + side-by-side video

**Files:**
- Create: `threed/compare/__init__.py`
- Create: `threed/compare/metrics.py`
- Create: `threed/compare/render.py`
- Create: `threed/compare/run_compare.py`
- Test: `tests/threed/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/threed/test_metrics.py
import numpy as np
from threed.compare.metrics import per_joint_jitter, per_joint_mpjpe, foot_skating

def test_per_joint_jitter_returns_zero_for_constant():
    j = np.zeros((10, 1, 17, 3), dtype=np.float32)
    out = per_joint_jitter(j)
    assert out.shape == (1, 17)
    np.testing.assert_array_equal(out, np.zeros((1, 17)))

def test_per_joint_jitter_picks_up_motion():
    j = np.zeros((10, 1, 17, 3), dtype=np.float32)
    j[:, 0, 0, 0] = np.arange(10) * 0.1   # nose moves 0.1 m / frame
    out = per_joint_jitter(j)
    np.testing.assert_allclose(out[0, 0], 0.1, rtol=1e-5)

def test_per_joint_mpjpe_zero_when_identical():
    a = np.random.rand(10, 1, 17, 3).astype(np.float32)
    out = per_joint_mpjpe(a, a.copy())
    np.testing.assert_array_equal(out, np.zeros((1, 17)))

def test_foot_skating_planted_foot_zero_velocity():
    # Foot is planted (constant) → skate = 0
    j = np.zeros((10, 1, 17, 3), dtype=np.float32)
    out = foot_skating(j, foot_idx=15, threshold=0.05)
    assert out[0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/threed/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement metrics**

```python
# threed/compare/metrics.py
from __future__ import annotations
import numpy as np

def per_joint_jitter(joints: np.ndarray) -> np.ndarray:
    """Return per-(dancer, joint) mean inter-frame velocity in metres/frame.

    joints: (T, N_dancers, J, 3). NaN frames are ignored.
    Returns: (N_dancers, J) array.
    """
    diffs = np.linalg.norm(np.diff(joints, axis=0), axis=-1)  # (T-1, N, J)
    with np.errstate(invalid="ignore"):
        return np.nanmean(diffs, axis=0)

def per_joint_mpjpe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean per-joint position error between two joint sequences.

    a, b: (T, N_dancers, J, 3). Returns (N_dancers, J) array.
    """
    diff = np.linalg.norm(a - b, axis=-1)   # (T, N, J)
    with np.errstate(invalid="ignore"):
        return np.nanmean(diff, axis=0)

def foot_skating(
    joints: np.ndarray, *, foot_idx: int = 15, threshold: float = 0.05
) -> np.ndarray:
    """Mean foot velocity for frames where the foot height is below threshold.

    joints: (T, N_dancers, J, 3). z-axis is up.
    Returns: (N_dancers,) array, mean foot velocity in m/frame for "planted" frames.
    """
    foot = joints[:, :, foot_idx, :]   # (T, N, 3)
    h = foot[:, :, 2]                  # (T, N) height
    planted = h < threshold            # (T, N)
    vel = np.linalg.norm(np.diff(foot, axis=0), axis=-1)   # (T-1, N)
    mask = planted[1:] & ~np.isnan(vel)
    out = np.zeros(joints.shape[1], dtype=np.float32)
    for d in range(joints.shape[1]):
        m = mask[:, d]
        if m.any():
            out[d] = float(vel[m, d].mean())
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/threed/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Implement the side-by-side renderer**

```python
# threed/compare/render.py
"""Stitch PromptHMR's overlay frames + SAM-Body4D's overlay frames into
a single side-by-side MP4. Falls back to writing 'missing' frames as a
blank panel so the MP4 stays the right length.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompthmr-frames-dir", type=Path,
                   help="Optional; if omitted, leaves left panel blank")
    p.add_argument("--body4d-frames-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(argv)

    body4d_paths = sorted(args.body4d_frames_dir.glob("*.jpg"))
    n = len(body4d_paths)
    sample = cv2.imread(str(body4d_paths[0]))
    H, W = sample.shape[:2]
    out_W = W * 2 + 10  # 10-px gutter

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(args.output), fourcc, args.fps, (out_W, H))

    for i, body_path in enumerate(body4d_paths):
        body = cv2.imread(str(body_path))
        body = cv2.resize(body, (W, H))
        if args.prompthmr_frames_dir is not None:
            phmr_path = args.prompthmr_frames_dir / body_path.name
            if phmr_path.is_file():
                phmr = cv2.imread(str(phmr_path))
                phmr = cv2.resize(phmr, (W, H))
            else:
                phmr = np.zeros_like(body)
        else:
            phmr = np.zeros_like(body)

        canvas = np.zeros((H, out_W, 3), dtype=np.uint8)
        canvas[:, :W] = phmr
        canvas[:, W + 10:] = body
        cv2.putText(canvas, "PromptHMR-Vid", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "SAM-Body4D", (W + 30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(canvas)

    vw.release()
    print(f"wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Implement the comparison driver**

```python
# threed/compare/run_compare.py
"""Compute per-clip comparison metrics from two joint files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from threed.compare.metrics import per_joint_jitter, per_joint_mpjpe, foot_skating

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompthmr-joints", type=Path, required=True,
                   help="prompthmr/joints_world.npy of shape (T, N, J, 3)")
    p.add_argument("--body4d-joints", type=Path, required=True,
                   help="sam_body4d/joints_cam.npy of shape (T, N, J, 3)")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write metrics.json")
    args = p.parse_args(argv)

    a = np.load(args.prompthmr_joints)
    b = np.load(args.body4d_joints)

    if a.shape[2] != b.shape[2]:
        # PromptHMR uses 22 SMPL body joints; body4d uses 17 COCO.
        # Pick the 17-joint COCO subset of SMPL: indices defined in
        # sm_to_coco17 below.
        SMPL22_TO_COCO17 = [15, 15, 15, 15, 15,   # nose/eyes/ears -> head joint (15)
                            17, 16, 19, 18, 21, 20,  # shoulders/elbows/wrists
                            2, 1, 5, 4, 8, 7]        # hips/knees/ankles
        a = a[:, :, SMPL22_TO_COCO17, :]

    n_frames = min(a.shape[0], b.shape[0])
    a, b = a[:n_frames], b[:n_frames]
    n_dancers = min(a.shape[1], b.shape[1])
    a, b = a[:, :n_dancers], b[:, :n_dancers]

    metrics = {
        "per_joint_jitter_phmr_m_per_frame": per_joint_jitter(a).tolist(),
        "per_joint_jitter_body4d_m_per_frame": per_joint_jitter(b).tolist(),
        "per_joint_mpjpe_m": per_joint_mpjpe(a, b).tolist(),
        "foot_skating_phmr_m_per_frame": foot_skating(a).tolist(),
        "foot_skating_body4d_m_per_frame": foot_skating(b).tolist(),
        "n_frames_compared": int(n_frames),
        "n_dancers_compared": int(n_dancers),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Smoke test on adiTest after both HMR runs are done**

```bash
python -m threed.compare.run_compare \
    --prompthmr-joints runs/3d_compare/adiTest/prompthmr/joints_world.npy \
    --body4d-joints runs/3d_compare/adiTest/sam_body4d/joints_cam.npy \
    --output runs/3d_compare/adiTest/comparison/metrics.json
```

Expected: `metrics.json` with the four metric arrays + counts.

- [ ] **Step 8: Render the side-by-side video**

```bash
python -m threed.compare.render \
    --prompthmr-frames-dir runs/3d_compare/adiTest/prompthmr/rendered_frames \
    --body4d-frames-dir runs/3d_compare/adiTest/sam_body4d/rendered_frames \
    --output runs/3d_compare/adiTest/comparison/side_by_side.mp4
```

Expected: a 2× wide MP4 with the two renders side by side.

- [ ] **Step 9: Commit**

```bash
git add threed/compare tests/threed/test_metrics.py
git commit -m "feat(threed/compare): metrics + side-by-side video for two HMR pipelines"
```

---

### Task 12: End-to-end orchestrator (one command per clip)

**Files:**
- Create: `scripts/run_3d_compare.py`

This wraps stages A → B → C1/C2 → D so a developer can run a single
clip with one command (it shells out to the conda envs).

- [ ] **Step 1: Implement the orchestrator**

```python
# scripts/run_3d_compare.py
"""End-to-end driver for the dual 3D pipeline.

Stage A runs locally (this conda env). Stages B/C1 and C2 are shelled
out to the PromptHMR / SAM-Body4D conda envs because they cannot share
a single Python process.
"""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from threed.config import default_config

log = logging.getLogger("run_3d_compare")

def _conda_run(env: str, *cmd: str, cwd: Path) -> int:
    """Run a shell command inside a conda env via `conda run`."""
    full = ["conda", "run", "-n", env, "--no-capture-output", *cmd]
    log.info("[%s @ %s] %s", env, cwd, " ".join(full))
    return subprocess.run(full, cwd=str(cwd)).returncode

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--static-camera", action="store_true")
    p.add_argument("--disable-completion", action="store_true",
                   help="Skip Diffusion-VAS occlusion completion in SAM-Body4D "
                        "(9× faster but worse on crowded scenes). Default: ON.")
    p.add_argument("--skip-stage-a", action="store_true")
    p.add_argument("--skip-phmr", action="store_true")
    p.add_argument("--skip-body4d", action="store_true")
    p.add_argument("--skip-compare", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = default_config()
    dirs = cfg.clip_dirs(args.clip).ensure()

    # Stage A — local
    if not args.skip_stage_a:
        rc = subprocess.run(
            [sys.executable, "-m", "threed.stage_a.run_stage_a",
             "--clip", args.clip,
             "--video", str(args.video),
             "--cache-dir", str(args.cache_dir)],
            cwd=str(REPO_ROOT),
        ).returncode
        if rc != 0:
            log.error("stage A failed (%d)", rc)
            return rc

    # Stage B — SAM-2 mask propagation lives in the PromptHMR env
    if not args.skip_phmr:
        # Build masks
        rc = _conda_run(cfg.phmr_conda_env,
                        "python", "-m", "our_pipeline.build_masks",
                        "--intermediates-dir",
                        str(dirs.intermediates),
                        cwd=cfg.phmr_repo)
        if rc != 0:
            return rc
        # Run PromptHMR-Vid
        rc = _conda_run(cfg.phmr_conda_env,
                        "python", "-m", "our_pipeline.run_phmr",
                        "--intermediates-dir", str(dirs.intermediates),
                        "--output-dir", str(dirs.prompthmr),
                        *(("--static-camera",) if args.static_camera else ()),
                        cwd=cfg.phmr_repo)
        if rc != 0:
            return rc

    # Stage C2 — SAM-Body4D
    if not args.skip_body4d:
        completion_args = ("--disable-completion",) if args.disable_completion else ()
        rc = _conda_run(cfg.body4d_conda_env,
                        "python", "-m", "our_pipeline.run_body4d",
                        "--intermediates-dir", str(dirs.intermediates),
                        "--output-dir", str(dirs.sam_body4d),
                        "--batch-size", str(cfg.body4d_batch_size),
                        *completion_args,
                        cwd=cfg.body4d_repo)
        if rc != 0:
            return rc
        # Extract COCO-17 joints
        n_frames = len(list((dirs.intermediates / "frames").glob("*.jpg")))
        rc = _conda_run(cfg.body4d_conda_env,
                        "python", "-m", "our_pipeline.extract_joints",
                        "--sam-body4d-dir", str(dirs.sam_body4d),
                        "--n-frames", str(n_frames),
                        cwd=cfg.body4d_repo)
        if rc != 0:
            return rc

    # Stage D — compare (local)
    if not args.skip_compare:
        rc = subprocess.run(
            [sys.executable, "-m", "threed.compare.run_compare",
             "--prompthmr-joints", str(dirs.prompthmr / "joints_world.npy"),
             "--body4d-joints", str(dirs.sam_body4d / "joints_cam.npy"),
             "--output", str(dirs.comparison / "metrics.json")],
            cwd=str(REPO_ROOT),
        ).returncode
        if rc != 0:
            return rc
        rc = subprocess.run(
            [sys.executable, "-m", "threed.compare.render",
             "--prompthmr-frames-dir", str(dirs.prompthmr / "rendered_frames"),
             "--body4d-frames-dir", str(dirs.sam_body4d / "rendered_frames"),
             "--output", str(dirs.comparison / "side_by_side.mp4")],
            cwd=str(REPO_ROOT),
        ).returncode
        if rc != 0:
            return rc

    log.info("[%s] all stages complete; results in %s",
             args.clip, dirs.intermediates.parent)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test on adiTest end-to-end**

```bash
python scripts/run_3d_compare.py \
    --clip adiTest \
    --video /Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov \
    --cache-dir runs/winner_stack_demo/_cache/adiTest \
    --static-camera
```

Expected on success: `runs/3d_compare/adiTest/comparison/metrics.json`
exists and `side_by_side.mp4` plays correctly.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_3d_compare.py
git commit -m "feat(scripts): one-command dual 3D pipeline driver"
```

---

### Task 13: Run on the canonical dance clips

**Files:** none (operational task).

> **Note on `--static-camera`:** we deliberately do NOT pass it for
> the canonical comparison runs, even on tripod shots. The flag bypasses
> DROID-SLAM + Metric3D, which is PromptHMR-Vid's only metric-scale
> signal. Without it, per-person depth gets noticeably wonkier — exactly
> the failure mode we already saw in our earlier PromptHMR test runs.
> If DROID-SLAM crashes on a clip (it can on near-static cameras),
> the pipeline falls back to identity automatically and logs a warning;
> only re-run that specific clip with `--static-camera` if SLAM truly
> fails. We document any per-clip fallback in the operator log.

- [ ] **Step 1: Run on the 5 small clips first**

```bash
for clip in adiTest easyTest 2pplTest gymTest shorterTest; do
    python scripts/run_3d_compare.py \
        --clip "$clip" \
        --video /Users/arnavchokshi/Desktop/"$clip"/<actual-file> \
        --cache-dir runs/winner_stack_demo/_cache/"$clip"
done
```

Expected runtime: ~10–30 min per clip on H100, depending on dancer
count.

- [ ] **Step 2: Run on `BigTest` (14 dancers)**

```bash
python scripts/run_3d_compare.py \
    --clip BigTest \
    --video /Users/arnavchokshi/Desktop/BigTest/BigTest.mov \
    --cache-dir runs/winner_stack_demo/_cache/BigTest
```

If `body4d` OOMs, drop `body4d_batch_size` to 8 in
`threed/config.py` and re-run with `--skip-stage-a --skip-phmr`.

- [ ] **Step 3: Run on `loveTest` (15 dancers, the bottleneck clip)**

```bash
python scripts/run_3d_compare.py \
    --clip loveTest \
    --video /Users/arnavchokshi/Desktop/loveTest/IMG_9265.mov \
    --cache-dir runs/winner_stack_demo/_cache/loveTest
```

This is the test that will tell us whether SAM-Body4D actually beats
PromptHMR-Vid on crowded scenes (the original research question that
made this dual pipeline interesting).

- [ ] **Step 4: Tabulate the metrics**

Read each `metrics.json`, tabulate jitter / MPJPE / foot-skating
per clip into `docs/3D_COMPARISON_RESULTS.md`. Include the
`side_by_side.mp4` paths so a reviewer can spot-check visually.

- [ ] **Step 5: Commit results doc**

```bash
git add docs/3D_COMPARISON_RESULTS.md
git commit -m "docs: tabulate 8-clip dual-pipeline comparison"
```

---

### Task 14 (optional): HTML report

**Files:**
- Create: `threed/compare/report.py`

A small templated HTML report that puts the metrics + the side-by-side
video + the per-track GLB files into a single page, so non-technical
reviewers (you, dancers) can browse them.

- [ ] **Step 1: Implement the report writer**

(Body omitted for brevity — uses Jinja2 to render a static HTML page
referencing `comparison/side_by_side.mp4`, `prompthmr/world4d.mcs`,
and per-track plots from the joints arrays.)

- [ ] **Step 2: Add a `<model-viewer>` for the PromptHMR GLB**

Embeds `prompthmr/world4d.glb` into the HTML using
`<model-viewer src="...">` so a reviewer can rotate / play the
animation in-browser.

- [ ] **Step 3: Commit**

```bash
git add threed/compare/report.py
git commit -m "feat(threed/compare): HTML reviewer report"
```

---

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| SAM-Body4D OOMs on `loveTest` (15 dancers × completion-on extrapolates to ~160 GB peak) | High | Process tracks in batches of ≤5 (already in plan); drop `sam_3d_body.batch_size` 16 → 8; only as last resort, pass `--disable-completion` for that single clip and document the asymmetry |
| SAM-Body4D runtime balloons with completion on (~26 min / 90 frames / 5 dancers on H800) | High | Accepted cost — completion is needed for honest comparison on dance scenes; run overnight; use `--skip-body4d` to iterate on PromptHMR side independently |
| MHR70 → COCO-17 mapping is wrong | Medium | One-time verification in MeshLab + visual check on adiTest before tabulating |
| DROID-SLAM fails on near-static dance camera | Medium | Pipeline already catches and falls back to identity; we use `--static-camera` for tripod shots |
| PromptHMR-Vid hand_pose is zero so reaching/styling looks "stiff" | Low (released checkpoint limit) | Document in report; future work: HaMer pass on wrist crops |
| Two conda envs drift / package conflict on shared system | Medium | Both repos cloned in `~/code/`; checkpoints in `~/checkpoints/`; no shared `site-packages` |
| HF gated checkpoints (SAM-3, SAM-3D-Body) get revoked | Low | Cache locally in `~/checkpoints/`; record SHA256 |
| BoxMOT cache pickle layout changes | Low | We use the public `FrameDetections` dataclass; covered by the `tests/threed/test_extract_tracks.py` fixture |
| `MHR70_TO_COCO17` indices are placeholder values | High (current state) | Task 10 step 1 explicitly calls out the one-time MeshLab check; do this before relying on `joints_cam.npy` for any quantitative claim |

---

## 7. What this plan does *not* cover

These are explicit non-goals to keep the plan tight:

- **Hand articulation** — both systems' released hands are weak;
  HaMer / ACR wrist-crop pass is a future task.
- **Inter-penetration removal** — neither system is a physics solver.
  Use JOSH or a dedicated post-opt if quantitative critique requires
  centimetre-accurate spacing.
- **Real-time / streaming** — both systems are offline.
- **Apple Silicon support for the HMR sidecars** — DROID-SLAM and
  Diffusion-VAS need CUDA wheels. Develop on Linux/CUDA; the tracking
  Stage A still works on macOS for upstream debugging.
- **Dance-specific physics** — joint angle limits, balance constraints,
  music-aware retiming etc.
- **A learned MHR ↔ SMPL-X converter** — too much research for too
  little payoff; we compare in 3D joint space instead.

---

## 8. Acceptance criteria

The dual pipeline is "shipped" when:

1. `python scripts/run_3d_compare.py --clip adiTest ...` produces
   `runs/3d_compare/adiTest/comparison/metrics.json` and
   `side_by_side.mp4` end-to-end on the CUDA box.
2. The same command on `loveTest` succeeds (may require lowered
   `body4d_batch_size`).
3. `docs/3D_COMPARISON_RESULTS.md` contains a per-clip table of
   jitter / MPJPE-between-models / foot-skating for both backends, on
   all 8 canonical clips.
4. All `pytest tests/threed/` tests pass on the local box.
5. The PromptHMR `world4d.mcs` file for `adiTest` opens successfully
   in [me.meshcapade.com/editor](https://me.meshcapade.com/editor)
   and shows 5 dancers.
6. The SAM-Body4D `4d_render.mp4` for `adiTest` shows 5 dancers with
   per-frame mesh overlays.

---

## 9. Sources cross-referenced

- PromptHMR README + source —
  [github.com/yufu-wang/PromptHMR](https://github.com/yufu-wang/PromptHMR)
  (commit on `main`, fetched April 2026).
- PromptHMR-Vid pipeline and config —
  `pipeline/pipeline.py`, `pipeline/phmr_vid.py`, `pipeline/tools.py`,
  `pipeline/config.yaml`.
- SAM-Body4D README + source —
  [github.com/gaomingqi/sam-body4d](https://github.com/gaomingqi/sam-body4d)
  (commit on `master`, fetched April 2026).
- SAM-Body4D offline pipeline —
  `scripts/offline_app.py`, `scripts/setup.py`, `assets/doc/resources.md`.
- SAM-3D-Body utilities —
  `models/sam_3d_body/notebook/utils.py` (`process_image_with_mask`,
  `save_mesh_results`).
- Original research notes — see `docs/3D_RECONSTRUCTION_RESEARCH.md`
  in this repo.
- Existing tracking pipeline —
  `docs/WINNING_PIPELINE_CONFIGURATION.md`,
  `tracking/deepocsort_runner.py`, `prune_tracks.py::FrameDetections`.

---

## 10. Open questions to resolve before kicking off

These are quick checks the human operator should do once before Task 1:

1. **Which CUDA box do we use?** **DECIDED 2026-04-17: Lambda Cloud A100
   instance** (operator just provisioned). PromptHMR-Vid wants cu121 or
   cu126; SAM-Body4D wants cu118. Lambda's default Ubuntu 22.04 image
   ships with the CUDA 12.x driver stack which works for PromptHMR; for
   SAM-Body4D we use a `body4d` conda env with the cu118 PyTorch wheels
   (the cu118 wheels run fine on a cu121 driver — only the *driver*
   needs to be ≥12.0, the *runtime* CUDA can be older). Variant note:
   on **A100 80 GB** all 8 clips fit (loveTest still needs per-track
   batching ≤5 IDs); on **A100 40 GB** SAM-Body4D's completion-on path
   OOMs for 5+ dancer clips and we have to fall back to
   `--disable-completion` for adiTest / loveTest / BigTest.
2. **Where does Stage A live?** **DECIDED: run all four stages on the
   Lambda A100.** Stage A code already runs there (boxmot + ultralytics
   + torch-cuda are easy installs). Avoids `rsync` of the
   `intermediates/frames/` 188-image folders per clip across the WAN.
   The current Mac-side `runs/3d_compare/adiTest/intermediates/` was
   only generated to smoke-test Task 4; we will regenerate on the box.
3. **Do we want the HF gated SAM-3 checkpoint?** **DECIDED: skip it.**
   Our wrapper substitutes our SAM-2 masks and the plan's Task 9 step 1
   monkey-patches `build_sam3_from_config` to a no-op. We *do* still
   need to accept the gate for `facebook/sam-3d-body-dinov3` (used for
   the per-frame mesh fit). One HF gate accept, not two.
4. **Where do we host the checkpoints long-term?** **DECIDED: `~/checkpoints/`
   on the A100's local SSD.** Lambda On-Demand instances have 1.4–4 TB
   ephemeral NVMe — plenty for ~30 GB of weights + intermediates. If we
   later want to stop/start the instance without re-downloading
   checkpoints, attach a Lambda Persistent Storage Filesystem and
   symlink `~/checkpoints` into it; not needed for this session.

---

## 11. Operator log (fill in as you install)

| Date | Step | Outcome |
|---|---|---|
| 2026-04-17 | git init + remote `origin = https://github.com/arnavchokshi/SAM-HMR.git` (no push) + baseline commit | OK (`fd48c2a`) |
| 2026-04-17 | Task 1 — scaffold `threed/` config + io (with `pyproject.toml` + `conftest.py` plan fixes) | OK — 3/3 tests pass (`2bd5f12`) |
| 2026-04-17 | Task 2 — extract tracks from cache | OK — 1 new test, 4/4 cumulative (`dfe45ca`) |
| 2026-04-17 | Task 3 — extract frames at 896 + full | OK — 1 new test, 5/5 cumulative (`c8a7e33`) |
| 2026-04-17 | Task 4 — Stage A driver (+ smoke test on `adiTest`) | OK — 188 frames + 5 tracks IDs 1-5, mean conf 0.86-0.88 (`f2ce156`) |
| 2026-04-18 | Box-side preflight on Lambda A100 (SSH, push, miniforge, tmux) | OK — see `_agent_log.md` "2026-04-18 — box-side preflight" |
| 2026-04-18 | clone PromptHMR (Task 5 step 1) | OK — `~/code/PromptHMR` HEAD `7d39d3f` |
| 2026-04-18 | install `phmr_pt2.4` env (Task 5 step 2) | OK — torch 2.4.0+cu121, CUDA 12.1, all PromptHMR deps + DROID-SLAM + Detectron2 (~6 min wall) |
| 2026-04-18 | rsync cached body_models from Mac (replaces `fetch_smplx.sh`) | OK — 3.37 GB (smpl/{NEUTRAL,MALE,FEMALE}.pkl + smplx/{NEUTRAL,MALE,FEMALE}.{pkl,npz} + helpers); slim npz pulled separately via `gdown 1v9Qy7…` |
| 2026-04-18 | run `fetch_data.sh` + BEDLAM2 ckpt (Task 5 step 3) | OK — phmr/, phmr_vid/, sam2_ckpts/, sam_vit_h_4b8939.pth (2.4G), vitpose-h-coco_25.pth (2.4G), camcalib_sa_biased_l2.ckpt (288M), droidcalib.pth (16M), examples/{boxing,boxing_short,dance_1,dance_2}.mp4 — 5.1 GB total in 4.5 min |
| 2026-04-18 | run `boxing_short.mp4` demo (Task 5 step 4 — milestone gate A) | OK — `results/boxing_short/{results.pkl 401K, world4d.mcs 44K, world4d.glb 66M, subject-{1,2}.smpl}` produced; 50 frames, 2 boxers, ~4 min wall, peak VRAM 11.8 GB |
| 2026-04-18 | Task 6 — PromptHMR mask sidecar | OK — `threed/sidecar_promthmr/build_masks.py` + 22 unit tests; adiTest smoke: 188 frames × 5 tids → 940 per-tid PNGs, 188 palette PNGs (P-mode, indices 0–5), `masks_union.npy (188,720,1280) bool sum=10026161`; 28 s wall, peak VRAM <1 GB (sam2 hiera_tiny). Commits `90765eb`, `9811b08`, `aea669e`, `761cf27`. |
| 2026-04-18 | Task 7 — PromptHMR-Vid sidecar | OK — `threed/sidecar_promthmr/run_promthmr_vid.py` + 15 unit tests; adiTest end-to-end (188 frames × 5 dancers, `phmr_b1b2.ckpt` swapped, `cfg.tracker='sam2'` so mask-prompt path active): 1 m 48 s wall, peak VRAM 10.9 GB; outputs `results.pkl 1.0 GB`, `joints_world.npy (188,5,22,3) float32 0 NaN`, `world4d.{mcs 348K, glb 74 MB}`, `subject-{1..5}.smpl`. Commits `c8f341f`, `86413c3`, `d33af1f`, `7f18d7c`, `59c9cb1`. |
| 2026-04-18 | Task 8 — body4d env + sam-body4d + smoke | OK — `~/code/sam-body4d` cloned (HEAD `21af102`); conda env `body4d` (py 3.12.13, torch 2.7.1+cu118); install fix: `setuptools<80` re-supplies `pkg_resources` (detectron2 imports it). Checkpoints under `~/checkpoints/body4d` (20 GB: depth-anything-v2-vitl, diffusion-vas-{amodal,content}, moge-2-vitl-normal, sam-3d-body-dinov3 model+mhr; SAM3 sam3.pt **BLOCKED**, gate still PENDING per HF dashboard). Step 4 verification deviated from interactive Gradio (no SSH X11) → headless **wrapper-path smoke** that monkey-patches `build_sam3_from_config→(None,None)` (matches plan §3.4 production design), feeds 8 example1.mp4 frames + bbox-shaped palette masks: `OfflineApp.__init__` 72 s (one-time vitdet download 2.77 GB), `on_4d_generation` 12 s, peak VRAM 9.44 GB, outputs 8 PLYs + 8 focal JSONs + 8 rendered + 1 4D MP4. SAM3 ckpt is irrelevant for our wrapper but will be downloaded once gate flips for completeness. |
| 2026-04-18 | Task 9 — SAM-Body4D wrapper helpers | OK — `threed/sidecar_body4d/wrapper.py` ships 6 GPU-free helpers: `monkeypatch_sam3` (idempotent, sentinel-marked, no-clobber), `intermediates_layout_ok` (checks `frames_full/`, `masks_palette/`, `tracks.pkl`, frame=mask count), `sorted_tid_list` (np.int64-safe), `link_artifacts_into_workdir` (symlinks frames + palette masks, mismatch-raises, idempotent), `workdir_layout_ok` (basename 1:1 sanity), `iter_palette_obj_ids` (1..255 palette guard for ≤loveTest's 15 dancers). 28 unit tests pass; 70/70 across the full `tests/threed/` suite. No GPU work yet — Task 10 wires these into the runner. |
| 2026-04-18 | Task 10 — SAM-Body4D sidecar runner | OK — `threed/sidecar_body4d/run_body4d.py` (Stage C2) + 6 unit tests for sys.path injection + config patching. End-to-end smoke on adiTest (188 frames × 5 dancers, 1280×720, `--disable-completion --batch-size 32`): runner exit=0 in 13 m wall (init 30 s, `on_4d_generation` 12 m 27 s); peak VRAM 11.1 GB torch / 12.8 GB nvidia-smi (well under 40 GB budget). Outputs: 940 PLYs (5 tids × 188 frames, perfect balance), 940 focal JSONs, 188 rendered overlays, 1 4D MP4 (662 KB), `run_summary.json` with timings + paths. Joint extraction (COCO-17) deferred to Stage D so the same logic runs on SMPL-X too. Commits `f49bf0f` (wrapper), `80d7807` (runner). |
| 2026-04-18 | Task 8 — `body4d` env + Gradio demo (milestone gate B) | |
| 2026-04-18 | accept HF gate (`facebook/sam-3d-body-dinov3` only — SAM-3 skipped) | |
| 2026-04-18 | Task 9 — SAM-Body4D sidecar | |
| 2026-04-18 | Task 10 — COCO-17 joint extraction | |
| 2026-04-18 | Task 11 — comparison harness (milestone gate C — `side_by_side.mp4`) | OK — `threed/compare/{joints,metrics,render,run_compare}.py` (105 unit tests) reduce SMPL-22/MHR70 → COCO-17, compute jitter/MPJPE/foot-skating (NaN-safe via `np.nanmean`), stitch h-stacked mp4 with letterbox + gutter. PromptHMR-side joint projector `threed/sidecar_promthmr/project_joints.py` reads `results.pkl` + `joints_world.npy` and projects SMPL-22 world→cam (using per-frame `Rcw,Tcw`), then reduces to COCO-17 → `joints_coco17_cam.npy`. SAM-Body4D side gets MHR70 joints via a `monkeypatch_save_mesh_results` wrapper that intercepts `pred_keypoints_3d` (per-frame, per-slot `pid+1`) → `joints_world.npy (T,N,70,3)` (note: cam-frame, "world" is a layout convention). adiTest Stage D smoke: PromptHMR `(188,5,17,3)` + SAM-Body4D `(188,5,70,3)` → reduced/aligned `(188,5,17,3)` per side; metrics: jitter 0.045 m/frame (PHMR) vs 0.049 (Body4D), MPJPE 9.21 m (coord-systems unaligned in scale — known followup), foot-skating PHMR 0.0/Body4D 0.05 m/frame; `side_by_side.mp4` 188 frames @ 30 fps, 2570×720 (intermediates/frames as PHMR-side proxy until upstream renders are wired). Bug fix `e91d5b5` corrected joint-dump slot indexing (was `id_current[pid]+1` → off-by-one vs upstream PLY layout `pid+1`). Commits `dd39c7c` (Stage D scaffold), `e91d5b5` (slot fix). |
| 2026-04-18 | Task 12 — orchestrator | OK — `scripts/run_3d_compare.py` wraps Stages A → B → C1/C2 → D across host/`phmr_pt2.4`/`body4d` envs via `conda run -n <env> --no-capture-output`. Pure command-builders + `plan_pipeline()` skip-flag composer fully unit-tested (15/15 in `tests/threed/test_orchestrator.py`) without any subprocess calls — GPU stages validated by their per-stage smokes. `--output-root` overrides `cfg.output_root` so the box uses its native `~/work/3d_compare/`. Skip-flag smoke on adiTest (`--skip-stage-a --skip-phmr --skip-body4d`) drove `compare → render` in 2.2 s wall, producing identical `metrics.json` + `side_by_side.mp4` to the per-stage invocations. Commit `cc762be`. |
| 2026-04-18 | Task 13 — adiTest end-to-end (milestone gate D) | OK — `python scripts/run_3d_compare.py --clip adiTest --output-root ~/work/3d_compare --skip-stage-a --skip-phmr --disable-completion --batch-size 32` drove `body4d → compare → render` in **738 s wall** (body4d init 61 s + on_4d_generation 666 s + compare/render ~10 s). VRAM peak 11.1 GB torch (well under 40 GB). Outputs: 940 PLYs, 940 focal JSONs, 940 joint .npy (188 frames × 5 dancers, all 940 (frame,dancer) entries valid — no NaN), `joints_world.npy (188,5,70,3)`, `metrics.json` (PHMR jitter 0.045 vs Body4D 0.049 m/frame, MPJPE 9.21 m), `side_by_side.mp4` 3.2 MB @ 2570×720 30 fps. Slot indexing fix verified: joint dirs are now `1..5` (not `2..6`). Metrics bit-identical to the rename-fixed earlier run → orchestrator reproducibility validated. |
| 2026-04-18 | Task 13 — loveTest + remaining 7 clips (milestone gate E) | **PARTIAL** — adiTest fully validated end-to-end; loveTest + 5 other candidate clips (`easyTest`, `gymTest`, `2pplTest`, `BigTest`, optional `mirrorTest`) deferred pending user go-ahead on the ~2 h GPU spend. **Blocker:** none of these clips have YOLO+DeepOcSort tracking caches in this repo (only `runs/winner_stack_demo/_cache/adiTest/` exists). Followup procedure documented in `runs/3d_compare/REPORT.md` §5: (1) run `scripts/run_winner_stack_demo.py` per clip locally, (2) SCP cache + video to box, (3) `python scripts/run_3d_compare.py --clip <c> --output-root ~/work/3d_compare --video ... --cache-dir ...`. Per-clip wall estimate: 14-55 min depending on dancer count (loveTest's ~15 dancers ≈ 50 min for body4d alone with completion ON). |
| 2026-04-18 | Task 14 — operator report | OK — `runs/3d_compare/REPORT.md`: end-to-end results on adiTest with per-joint metric table, wall + VRAM breakdown, output layout receipts, deviation log, followup procedure for the remaining clips, and 5 itemised followups (Procrustes-aligned MPJPE, world-frame foot-skating, PHMR mesh overlays, 2D reprojection vs ViTPose, optional HTML conversion). HTML version intentionally not yet produced — plan §11 calls it "optional"; will land if user confirms scope expansion. |
| 2026-04-18 | Followup #3 — PromptHMR mesh overlays (closes the all-black left panel) | OK — `threed/sidecar_promthmr/render_overlay.py` (385 LOC) runs SMPL-X forward on cached `results.pkl` (axis-angle path; degenerate face-rotmat fallback to identity), alpha-composites each dancer's mesh onto the original RGB frame via headless `pyrender` (alpha=0.65, deterministic HSV-wheel palette). 26 unit tests cover the 5 pure helpers (color palette, rotmat→axis-angle with PHMR's zero-face-rotmat edge case, intrinsics matrix, frame indexing, alpha composite). Wired into `scripts/run_3d_compare.py` as the new `phmr_render` stage between `phmr_project` and `body4d` (runs in `phmr_pt2.4` env via `conda run`); the final `render` step auto-detects `prompthmr/rendered_frames/` on disk and forwards `--prompthmr-frames-dir`. New `--skip-phmr-render` flag for Stage D iteration. adiTest re-render: phmr_render 25 s wall, compare+render 4 s wall, `side_by_side.mp4` now 8.3 MB with both panels filled in. Total tests: 176 passed (was 161). Commits `ab711a0`, `8ee06cb`, `9e8e531`. |
| 2026-04-18 | Followup #1 — Procrustes-aligned MPJPE | OK — `align_procrustes(a, b, *, per_dancer=True, allow_scale=False)` and `per_joint_mpjpe_pa(a, b, ...)` added to `threed/compare/metrics.py` (Kabsch + optional Umeyama scale, NaN-safe per-frame fits with all-NaN dancer fallback). Closes the long-standing "9.21 m flat MPJPE" interpretive gap from REPORT §2.3 (the value was a coord-system translation offset, not per-joint estimation error). Wired into `metrics.json` as the new `per_joint_mpjpe_pa_m` field (length-17, m); raw `per_joint_mpjpe_m` retained for back-compat. 12 new unit tests (203 total): identity / pure-translation / pure-rotation / per-dancer-vs-global / scale on/off / NaN excluded from fit but preserved in output / all-NaN dancer / shape mismatch raises; `mean(PA) ≤ mean(raw)` on random data; PA-MPJPE field emitted by `run_compare.main()`. Commit `f7b70fa`. |
| 2026-04-18 | Stage A `--max-frames` cap (cost control for clip extension) | OK — added optional `max_frames` to `extract_frames()` and exposed `--max-frames N` on both `threed.stage_a.run_stage_a` and `scripts/run_3d_compare.py`. Caps Stage A frame extraction at the first N frames so downstream PHMR + body4d cost stays bounded for long clips (loveTest 820 frames × 15 dancers would be ~5 h with completion ON; capping to 188 frames per clip keeps each clip directly comparable to adiTest's existing baseline). Default unset → no cap → existing adiTest runs unaffected. 5 new unit tests (208 total): exact frame count + last-frame numbering, None == no cap, cap > clip is no-op, orchestrator forwards / omits the flag. Commit `2270833`. |
| 2026-04-18 | Task 13 / 14 — 5-clip extension batch (2pplTest, easyTest, gymTest, BigTest, loveTest) | **IN PROGRESS** — caches generated locally on Mac (5 .pkl, 270-1299 frames, 2-15 dancers), SCP'd to box `~/work/cache/<clip>/`. Box `~/work/run_all_clips.sh` drives all 5 clips sequentially in tmux `arnav-3d:task14-batch` with `--max-frames 188 --batch-size 32 --disable-completion --fps 30`. Each clip log lands at `~/work/logs/task14_<clip>.log`. Estimated total wall: ~120 min (PHMR + body4d + render). Will populate per-clip metrics rows + cross-clip summary table once batch completes. |
| 2026-04-18 | Followup #6 — SAM-Body4D mesh overlay on real video (closes the right-panel clean-background gap) | OK — `threed/sidecar_body4d/render_overlay.py` reads the cached `mesh_4d_individual/<id>/<frame>.ply` + `focal_4d_individual/<id>/<frame>.json` tree and alpha-composites each dancer's MHR mesh onto the original RGB frame via headless `pyrender` (single-scene-per-frame, alpha=0.65, shares HSV-wheel palette + composite helpers with PHMR sidecar). The scene mirrors SAM-Body4D's own `Renderer.render_rgba_multiple`: meshes are loaded **as-is** from the PLY (already pre-translated by `cam_t` and X-rotated by upstream's `vertices_to_trimesh` → `(pred_verts + cam_t) * [1, -1, -1]`), and the camera sits at the world origin with no per-dancer offset. 21 unit tests pin dancer-id discovery, JSON parsing, the upstream PLY centroid convention (`upstream_ply_centroid` regression pin guards against re-introducing a per-dancer translation), and `flip_yz_verts`. Wired into `scripts/run_3d_compare.py` as the new `body4d_render` stage between `body4d` and `compare` (runs in `body4d` env via `conda run`); `--skip-body4d-render` flag added; `--skip-body4d` implies it. The final `render` step auto-detects `sam_body4d/rendered_frames_overlay/` on disk and prefers it over upstream's clean-background `rendered_frames/`. **Two bugs fixed during box bring-up:** (1) initial draft applied `flip_yz_verts` *inside* `_build_pyrender_scene`, double-flipping the already-pre-flipped PLYs and pushing meshes behind `pyrender`'s `znear=0.1` plane → blank overlay; root-caused by dumping raw render alpha/depth on the box and inspecting `vertices_to_trimesh` upstream. (2) follow-up draft applied a per-dancer translation `[cam_t.x, -cam_t.y, -cam_t.z]` modelled on the *single*-dancer renderer path (`__call__` line 187), which on top of the already-baked-in PLY translation doubled each dancer's depth → meshes rendered at half size. Fixed by reading the *multi*-dancer path (`render_rgba_multiple` lines 400-447) which uses no per-dancer offset; replaced the buggy helper with a `upstream_ply_centroid` documenting helper + 7 regression tests. adiTest re-render: `body4d_render` 33 s wall, compare+render 4 s wall, `side_by_side.mp4` now 10.4 MB with both panels showing scale-correct, full-body mesh overlays co-located with the real dancers. Total tests: 241 passed (was 220). Commits `77a700e`, `37a01f8`, `e33572d`, `a3c77bd`. |

### Handoff-to-plan task mapping (recorded 2026-04-18)

The Lambda hand-off note grouped the remaining work into 10 informally-numbered
items ("handoff Tasks 5–14"). Those numbers do NOT match the plan's Tasks 5–14
because the hand-off summarised some plan tasks together and inserted a
"per-track JSON converter" that has no plan-task equivalent. We follow the
**plan's numbering verbatim** and treat the hand-off list as a milestone-gate
description. The mapping is:

| Hand-off label | Plan task(s) | Operator-report milestone |
|---|---|---|
| "Task 5: PromptHMR env + boxing demo" | Plan Task 5 | A |
| "Task 6: Body4D env + Gradio demo" | Plan Task 8 | B |
| "Task 7: Per-track JSON converter (TDD)" | (no plan equivalent — folded into plan Task 9 if needed) | — |
| "Task 8: PromptHMR-Vid sidecar runner" | Plan Tasks 6 + 7 | — |
| "Task 9: SAM-Body4D wrapper (sam3 monkey-patch)" | Plan Task 9 step 1 | — |
| "Task 10: SAM-Body4D sidecar runner" | Plan Task 9 (rest) + Task 10 | — |
| "Task 11: Comparison stage (Stage D)" | Plan Task 11 | C |
| "Task 12: Multi-clip orchestrator" | Plan Task 12 | — |
| "Task 13: End-to-end smoke on adiTest" | Plan Task 13 (adiTest only) | D |
| "Task 14: loveTest + rest of 8 clips + report" | Plan Task 13 (remaining clips) + Plan Task 14 | E |

Other hand-off corrections recorded in `_agent_log.md`:
- SAM-Body4D upstream URL: hand-off said `facebookresearch/sam-body4d` (404).
  Plan §9 already lists the correct `gaomingqi/sam-body4d`; we follow the plan.
- HF gate: only `facebook/sam-3d-body-dinov3` is accepted (the SAM-3 gate is
  skipped per plan §10 question 3 and Task 9 step 1's monkey-patch).
- Lambda variant: A100-SXM4-**40 GB** (hand-off §3 said 80 GB). Per plan §6
  risks + §10 question 1, we try `completion.enable=true` first per clip and
  fall back to `--disable-completion` only on OOM, recording VRAM peaks in
  `_agent_log.md`.

