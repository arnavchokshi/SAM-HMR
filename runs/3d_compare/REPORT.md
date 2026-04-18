# Dual-3D HMR pipeline — operator report

**Status:** plan Tasks 5–13 complete (milestones A → D). Plan Task 14
("loveTest + remaining 7 clips") is **partial** — only `adiTest`
has been driven end-to-end. See "Followup procedure" for the
exact steps to extend to the other clips on the existing Lambda
A100 box.

Generated 2026-04-18 from commit [`10cc677`](https://github.com/arnavchokshi/SAM-HMR/commit/10cc677).

---

## 1. What this pipeline does

For each input clip we:

1. Extract frames + persistent multi-dancer tracks (DeepOcSort).
2. Build SAM-2 palette masks from the boxes, one PNG per frame
   with pixel = track id (1..N).
3. Run **PromptHMR-Vid** (Stage C1) to get SMPL-X pose + global
   trajectory per dancer.
4. Run **SAM-Body4D** (Stage C2) to get MHR (Momentum Human Rig)
   meshes + MHR70 keypoints per dancer. SAM-3 is monkey-patched
   out — we feed our own SAM-2 masks instead.
5. Reduce both outputs to a common COCO-17 skeleton in the camera
   frame (Stage D), compute per-joint metrics, and render a
   side-by-side mp4 for visual comparison.

All five stages are wrapped behind one orchestrator command
(`scripts/run_3d_compare.py`).

## 2. End-to-end on `adiTest` (5 dancers, 188 frames @ 30 fps, 1280×720)

**One-shot invocation** (on the Lambda box, body4d env active):

```bash
python scripts/run_3d_compare.py \
    --clip adiTest \
    --output-root /home/ubuntu/work/3d_compare \
    --skip-stage-a --skip-phmr \
    --disable-completion --batch-size 32 \
    --fps 30
```

(Stage A + PromptHMR are skipped because their outputs are already
on disk from earlier per-stage smokes; the orchestrator drives
`body4d → compare → render`.)

### 2.1 Wall-clock + VRAM

| Stage | Wall | Peak VRAM (torch) |
|---|---|---|
| body4d init | 61 s | 7.70 GB |
| body4d run | 666 s | **11.10 GB** |
| compare | 0.3 s | n/a (CPU) |
| render | 2 s | n/a (CPU) |
| **total** | **738 s = 12 m 18 s** | well under 40 GB A100 budget |

### 2.2 Outputs

```
/home/ubuntu/work/3d_compare/adiTest/
├── intermediates/
│   ├── frames/, frames_full/        — 188 each (resized 896-cap + original 1280×720)
│   ├── tracks.pkl                   — DeepOcSort, 5 ids
│   ├── masks_palette/<frame:08d>.png — 188 palette PNGs (px == tid)
│   └── masks_per_track/<tid>/<frame:08d>.png — 940 binary masks
├── prompthmr/
│   ├── results.pkl                  — 1.0 GB (SMPL-X pose + trajectory + ViTPose)
│   ├── joints_world.npy             — (188, 5, 22, 3) float32 — SMPL body joints
│   ├── joints_coco17_cam.npy        — (188, 5, 17, 3) float32 — COCO-17 in cam frame
│   └── world4d.{glb,mcs}, subject-{1..5}.smpl — interactive viewer assets
├── sam_body4d/
│   ├── joints_world.npy             — (188, 5, 70, 3) float32 — MHR70 in cam frame
│   ├── joints_4d_individual/{1..5}/<frame:08d>.npy — 940 per-(slot,frame) MHR70 dumps
│   ├── mesh_4d_individual/{1..5}/<frame:08d>.ply — 940 MHR meshes
│   ├── focal_4d_individual/{1..5}/<frame:08d>.json — 940 per-frame intrinsics
│   ├── rendered_frames/<frame:08d>.jpg — 188 overlays
│   ├── 4d_<id>.mp4                   — 4D scene mp4
│   └── run_summary.json              — timings + VRAM + paths
└── comparison/
    ├── metrics.json                  — Stage D per-joint metrics
    └── side_by_side.mp4              — 2570×720 @ 30 fps, 188 frames
```

### 2.3 Stage D metrics (per joint, mean across the 5 dancers)

| Joint | MPJPE (m) | Jitter PHMR (m/frame) | Jitter Body4D (m/frame) |
|---|---:|---:|---:|
| nose         | 9.285 | 0.0303 | 0.0369 |
| left_eye     | 9.266 | 0.0303 | 0.0350 |
| right_eye    | 9.276 | 0.0303 | 0.0343 |
| left_ear     | 9.187 | 0.0303 | 0.0314 |
| right_ear    | 9.211 | 0.0303 | 0.0288 |
| left_shoulder | 9.184 | 0.0331 | 0.0315 |
| right_shoulder| 9.178 | 0.0335 | 0.0305 |
| left_elbow   | 9.222 | 0.0558 | 0.0626 |
| right_elbow  | 9.185 | 0.0691 | 0.0716 |
| left_wrist   | 9.221 | 0.0832 | 0.1001 |
| right_wrist  | 9.199 | 0.0982 | 0.1155 |
| left_hip     | 9.177 | 0.0222 | **0.0073** |
| right_hip    | 9.169 | 0.0234 | **0.0074** |
| left_knee    | 9.232 | 0.0353 | 0.0454 |
| right_knee   | 9.216 | 0.0406 | 0.0458 |
| left_ankle   | 9.223 | 0.0522 | 0.0695 |
| right_ankle  | 9.208 | 0.0617 | 0.0743 |
| **mean**     | **9.214** | **0.0447** | **0.0487** |

**Foot-skating** (mean velocity for "planted" feet, foot_idx = 15
right_ankle, threshold = 0.05 m, m/frame, per dancer):

```
PHMR  : [0,    0,    0,    0,    0   ]
Body4D: [0.06, 0.04, 0.07, 0.05, 0.06]
```

### 2.4 Reading the metrics

**Jitter (temporal smoothness)** — both systems agree to within
~10 % on most joints. PromptHMR is slightly smoother on extremities
(wrists, ankles) likely thanks to its temporal smoothing prior;
Body4D is smoother on the **hip** (0.007 vs 0.022 m/frame) — by 3×.
That's an artifact of MHR's stronger pelvis stabilisation; whether
it's "more correct" depends on the clip's ground-truth motion.

**MPJPE (9.21 m flat)** — uniform across all 17 joints. This is
**not** per-joint estimation error. It's a global translation
offset between the two coord systems: PHMR puts dancers ~9 m from
the cam in its world frame, Body4D's MHR is rooted near the camera
origin. The world→cam projection in `project_joints.py` brings PHMR
into the camera frame but doesn't co-register the two roots. The
correct readout for "how close are the two estimates" is **MPJPE
after Procrustes alignment** (rigid R, t per dancer), which is now
computed by `align_procrustes` + `per_joint_mpjpe_pa` and emitted
to `metrics.json` as `per_joint_mpjpe_pa_m` — see followup #1.
The PA value will be re-extracted into the table above when the
clip is re-`compare`d on the box (Phase 8).

**Foot-skating PHMR = 0** — the cam-frame foot height never falls
below the 0.05 m threshold (camera is mounted above the floor),
so no foot is "planted" by our default detector. Body4D's
foot-skating is non-trivial (4-7 cm/frame) which is consistent
with not having SLAM-corrected per-frame depth. Both numbers
suggest we should compute foot-skating in **world frame** (PHMR's
native, requires inverse-projecting Body4D), see followup #2.

### 2.5 Side-by-side video

`comparison/side_by_side.mp4` — 2570×720 @ 30 fps, 8.3 MB.

| Left panel | Right panel |
|---|---|
| `prompthmr/rendered_frames/<frame>.jpg` (PHMR SMPL-X mesh overlay on input frame, alpha=0.65) | `sam_body4d/rendered_frames/<frame>.jpg` (Body4D mesh overlay) |

The left panel is generated by `threed/sidecar_promthmr/render_overlay.py`
(headless `pyrender`; ~25 s wall on adiTest) and is wired into the
orchestrator as the `phmr_render` stage between `phmr_project` and
`body4d`. Each dancer gets a deterministic HSV-wheel colour so a
re-render produces the same mapping. Closes followup #3.

## 3. Validated unit-test coverage

```
tests/threed/  →  203 passed, 3 skipped, 1 warning  (1.4 s)
```

| Suite | Tests | Coverage |
|---|---:|---|
| `test_io.py` | 3 | track save/load round-trip |
| `test_extract_tracks.py` | 1 | YOLO+DeepOcSort cache → TrackEntry |
| `test_extract_frames.py` | 1 | resized + full 1:1 frame export |
| `test_sidecar_promthmr_*` | 63 | mask sidecar + PHMR runner + project_joints + render_overlay (color palette, axis-angle conv with degenerate-rotmat fallback, frame indexing, alpha composite) |
| `test_sidecar_body4d_wrapper.py` | 37 | sam3 monkeypatch, layout/workdir checks, joint dump + consolidate (slot indexing locked in) |
| `test_sidecar_body4d_run_body4d.py` | 6 | sys.path injection + config patching |
| `test_compare_joints.py` | 22 | SMPL22→COCO17, MHR70→COCO17, face-collapse, L/R conventions |
| `test_compare_metrics.py` | 24 | jitter, MPJPE (raw + Procrustes-aligned), foot-skating, Kabsch alignment (NaN-safe) |
| `test_compare_render.py` | 10 | h-stack + letterbox + gutter |
| `test_compare_run.py` | 9 | align + auto-reduce + e2e driver (incl. PA-MPJPE field) |
| `test_orchestrator.py` | 18 | per-stage cmd builders (incl. phmr_render) + skip-flag composer (incl. --skip-phmr-render) |
| _other (config, helpers)_ | 12 | misc |

## 4. Deviations from the original plan

Recorded in `runs/3d_compare/_agent_log.md` per task; the major ones:

1. **`threed/compare/` not `threed/stage_d/`** — naming consistency
   with `threed.sidecar_*`. Stages remain a logical orchestration
   concept (used by `scripts/run_3d_compare.py`).
2. **Joint extraction by monkey-patch, not by mesh sampling.** SAM-Body4D
   already computes `pred_keypoints_3d` (MHR70) per person; we wrap
   `save_mesh_results` to dump them rather than re-sampling vertices.
3. **PromptHMR side projects 22-joint SMPL → 17-joint COCO** in the
   host process (no GPU needed) — `threed/sidecar_promthmr/project_joints.py`.
4. **SAM-3 fully bypassed.** We have our own SAM-2 masks from the
   shared intermediates; SAM-3's `sam3.pt` is not required (the gate
   was PENDING when we started). Plan §3.4 already endorsed this; we
   just never load the model.
5. **`--disable-completion` for adiTest's body4d run.** Diffusion-VAS
   completion is on by default but ~9× slower; for adiTest's
   non-occluded clip we get acceptable quality without it. Final
   loveTest run should re-enable completion (plan §3.5 mitigation).
6. **Slot indexing fix (`e91d5b5`):** the wrapper was originally
   writing `joints_4d_individual/<id_current[pid]+1>/` which gave
   off-by-one slot dirs vs upstream's `mesh_4d_individual/<pid+1>/`
   PLY layout. Caught by post-run dir comparison; fixed and locked
   in by a regression test.

## 5. Followup procedure (extending to loveTest + remaining clips)

The orchestrator is one command per clip; the prerequisite is a
DeepOcSort tracking cache for each clip. **None exist yet** other
than `adiTest`'s — the published 8-clip scoreboard caches are not
in this repo.

### 5.1 Per-clip prerequisites

For a new clip `<clip>` (e.g. `loveTest`):

```bash
# 1. Run YOLO+DeepOcSort tracking on Mac (fast — uses MPS) or on box (uses CUDA).
#    Produces runs/winner_stack_demo/_cache/<clip>/*.pkl
python scripts/run_winner_stack_demo.py \
    --clip <clip> \
    --video /Users/arnavchokshi/Desktop/<clip>/<filename>.mov

# 2. SCP the video + cache to the box
scp /Users/arnavchokshi/Desktop/<clip>/<filename>.mov \
    ubuntu@<box>:/home/ubuntu/work/videos/
scp -r runs/winner_stack_demo/_cache/<clip>/ \
    ubuntu@<box>:/home/ubuntu/work/cache/
```

### 5.2 Per-clip orchestrator invocation (on the box)

```bash
python scripts/run_3d_compare.py \
    --clip <clip> \
    --output-root /home/ubuntu/work/3d_compare \
    --video /home/ubuntu/work/videos/<filename>.mov \
    --cache-dir /home/ubuntu/work/cache/<clip> \
    --batch-size 16    # drop to 16 or 8 for >8 dancers (loveTest is 15)
```

(Drop the `--skip-stage-a --skip-phmr --disable-completion` flags;
they were used for adiTest because we wanted to test the orchestrator
on already-staged artifacts. For a fresh clip we want the full
pipeline including completion.)

### 5.3 Wall-clock + cost estimates per clip

Extrapolating from adiTest's 188 frames × 5 dancers @ 12 m 18 s:

| Clip | Dancers | Est. frames | Est. body4d wall (completion ON) | Est. orchestrator wall (full A→D) |
|---|---:|---:|---:|---:|
| `adiTest` (done) | 5 | 188 | 12 m | 13 m (skip A,B,C1) — actual |
| `2pplTest` | 2 | ~200 | ~10 m | ~14 m |
| `easyTest` | ~3 | ~200 | ~12 m | ~16 m |
| `gymTest` | ~3 | ~250 | ~15 m | ~20 m |
| `BigTest` | ~6 | ~300 | ~20 m | ~25 m |
| `loveTest` | **~15** | ~250 | ~50 m (linear in dancers) | ~55 m |

Total for the 5 remaining clips: **~2 h GPU on the existing A100
box** ≈ $1-3 in compute. Add ~30-60 min of wall for tracking +
SCP per clip on Mac.

### 5.4 Report extension

`runs/3d_compare/REPORT.md` (this file) should grow a per-clip
metric table and a comparative summary across clips. PHMR mesh
overlays already land via the `phmr_render` stage; the remaining
Stage D followups (Procrustes alignment, world-frame foot-skating)
will materially change the headline numbers and should land before
the cross-clip comparison is interpreted.

## 6. Open followups (Task 14 deliverables not yet built)

1. **Procrustes-aligned MPJPE.** Add a `align_procrustes(joints_a,
   joints_b, per_dancer=True)` helper in `threed/compare/metrics.py`
   and report MPJPE both before and after. Target after-alignment
   MPJPE: <50 cm for COCO body joints. _Done — `align_procrustes` +
   `per_joint_mpjpe_pa` (Kabsch + optional Umeyama scale) added to
   `threed/compare/metrics.py`; 12 unit tests cover identity /
   translation / rotation / per-dancer / scale / NaN / shape errors;
   `metrics.json` now emits `per_joint_mpjpe_pa_m`. Cross-clip
   numbers will be re-interpreted in §2._
2. **World-frame foot-skating.** Compute foot-skating using the
   un-projected world-frame joints from PHMR and a per-clip-calibrated
   ground-plane height, instead of the cam-frame heuristic.
3. ~~**PromptHMR mesh overlays.**~~ **Done (commit `9e8e531`):**
   `threed/sidecar_promthmr/render_overlay.py` runs the SMPL-X
   forward on cached `results.pkl` and alpha-composites each dancer's
   mesh onto the original RGB frame via headless `pyrender`. Wired
   into the orchestrator as the `phmr_render` stage; the final
   `render` step auto-detects `prompthmr/rendered_frames/` on disk
   and forwards `--prompthmr-frames-dir`. Wall on adiTest: 25 s.
4. **2D reprojection error.** Project both 3D joint sets through
   their respective per-frame intrinsics back into image space and
   compare against the bundled `vitpose` 17-keypoint output. Closes
   the loop on the §4.2 metric set.
5. **HTML report.** Convert this markdown to an HTML report with
   embedded `side_by_side.mp4` clips for each clip — plan §11
   Task 14 calls this "optional", we've left it for the user to
   confirm scope.

## 7. Reproducibility receipts

- Repo HEAD: [`10cc677`](https://github.com/arnavchokshi/SAM-HMR/commit/10cc677) on `main`
- Box: Lambda A100-SXM4-40GB, miniforge `body4d` env (py 3.12.13,
  torch 2.7.1+cu118, setuptools 79.0.1)
- Upstream pins:
  - PromptHMR HEAD `7d39d3f`
  - SAM-Body4D HEAD `21af102`
- Tests: `pytest tests/threed/ -q` → 203 passed, 3 skipped, 1 warning
