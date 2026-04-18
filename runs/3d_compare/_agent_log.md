# Agent log — dual 3D HMR pipeline build

Operator: AI coding agent (Cursor / Opus 4.7)
Plan: `docs/3D_DUAL_PIPELINE_PLAN.md`
Scope this session: Tasks 1, 2, 3, 4 (Stage A — runs in the existing tracking env).
Tasks 5–14 are CUDA-only and require the cloned PromptHMR / SAM-Body4D repos
and their conda envs; out of scope for this session.

## Session start
- Started: 2026-04-17 23:28 EDT

## Pre-flight findings (Step 2 verifications)
- `prune_tracks.py` with `FrameDetections` dataclass: present, matches plan.
- `tracking/deepocsort_runner.py`: present, exports `run_deepocsort` returning `RawTrack[]`.
- `threed/` and `tests/threed/`: do **not** exist yet (correct — this session creates them).
- Active conda env: `base` (Python 3.11.7) — plan calls it "tracking" but the
  active env is just `base`. Conda base **has pytest 8.4.1, numpy 1.26.4,
  cv2 4.11.0, joblib 1.5.1** — sufficient for Tasks 1-3 tests and Task 4 driver.
- Project also ships a `.venv/` (Python 3.11.0) which has boxmot 18.0.0 +
  ultralytics 8.4.37 + torch 2.11.0 but **no pytest**. Used only for generating
  the tracking cache (Task 4 prep) since boxmot is not in conda base.
- `runs/winner_stack_demo/_cache/adiTest/`: did **not** exist at session start
  — generated below as Task 4 prep before the smoke test.
- adiTest video: `/Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov` ✓
- YOLO weights: `weights/best.pt` ✓
- All 8 canonical test clips exist on `~/Desktop/` (adiTest, BigTest,
  easyTest, gymTest, loveTest, mirrorTest, shorterTest, 2pplTest).
- Open questions in §10 of the plan: none block Tasks 1-4.

## Deviations from the plan
- **Repo was not a git repository at session start**; the plan's per-task
  `git commit` steps assume git exists. Initialized with `git init -b main`,
  added `https://github.com/arnavchokshi/SAM-HMR.git` as `origin` (no push,
  per user constraint), and committed the existing source as a baseline
  before starting Task 1. This was discussed with the user and is the only
  out-of-band action taken before Task 1.
- The plan's environment is named `tracking` in the docs but in this checkout
  it is conda `base` + a `.venv/`. No env is named `tracking`. This is a doc
  drift, not a bug.

## Task progress

| Task | Status | Notes |
|---|---|---|
| Baseline `git init` + commit | DONE | `fd48c2a` chore: import existing tracking pipeline + plan |
| Plan fix #1 (pyproject.toml note) | DONE | `3dbbe95` docs(plan): note pyproject.toml pytest config required |
| Plan fix #2 (drop tests/threed/__init__.py, use importlib) | DONE | `508c230` docs(plan): drop tests/threed/__init__.py, use importlib import-mode |
| Task 1 — scaffold `threed/` config + io | **PASS** | `2bd5f12` feat(threed): scaffold dual-pipeline config + tracks IO. 3/3 tests pass. |
| Task 2 — extract tracks from cache | **PASS** | `dfe45ca` feat(threed/stage_a): extract per-track dict from DeepOcSort cache. 1 new test, 4/4 cumulative. |
| Task 3 — extract frames at 896 + full | **PASS** | `c8a7e33` feat(threed/stage_a): extract frames at 896 + full resolution. 1 new test, 5/5 cumulative. |
| Task 4 step 1 — Stage A driver | **PASS** | Driver written at `threed/stage_a/run_stage_a.py`. |
| Task 4 prep — generate adiTest cache | **PASS** | Ran `scripts/run_winner_stack_demo.py --clips adiTest --device mps --skip-render` from `.venv` in 35s. Cache: `runs/winner_stack_demo/_cache/adiTest/imgsz768_conf0.310_iou0.700_boxmot_deepocsort_457d921a09e4_max188.pkl`. |
| Task 4 step 2 — smoke test | **PASS** | Driver wrote 188 frames + 5-track tracks.pkl into `runs/3d_compare/adiTest/intermediates/`. All 5 dancers (IDs 1-5) have full 188-frame coverage, mean confs 0.86-0.88. |
| Task 4 step 3 — commit driver | **PASS** | `f2ce156` feat(threed/stage_a): bboxes + frames driver. |

## Open questions for the user
- (none currently)

## Commits this session
```
f2ce156 feat(threed/stage_a): bboxes + frames driver
c8a7e33 feat(threed/stage_a): extract frames at 896 + full resolution
dfe45ca feat(threed/stage_a): extract per-track dict from DeepOcSort cache
2bd5f12 feat(threed): scaffold dual-pipeline config + tracks IO
508c230 docs(plan): drop tests/threed/__init__.py, use importlib import-mode
3dbbe95 docs(plan): note pyproject.toml pytest config required for Task 1
fd48c2a chore: import existing tracking pipeline + 3D dual-pipeline plan
```

## Final state (end of Tasks 1-4)
- **All Stage A code in `threed/`** runs from conda `base` (no boxmot/torch deps).
- **Generating tracking caches** uses the project's `.venv/` (boxmot 18.0.0,
  ultralytics 8.4.37, torch 2.11.0, mps); already done for `adiTest`.
- **All Stage A artifacts for adiTest** live at
  `runs/3d_compare/adiTest/intermediates/`:
  - `frames/` — 188 JPEGs (PromptHMR-Vid input, max-height 896; here 720p
    untouched because the source is 720p)
  - `frames_full/` — 188 JPEGs at original resolution (SAM-Body4D input)
  - `tracks.pkl` — joblib dict `{tid: TrackEntry(...)}` for the 5 dancers
- **Test suite**: `pytest tests/threed/` from conda `base` reports 5/5 pass.
- **GitHub remote** `origin = https://github.com/arnavchokshi/SAM-HMR.git` is
  configured but **not pushed** (per user constraint). Run `git push -u origin main`
  manually when ready.

## What needs to be set up for Tasks 5-14 (out of scope this session)
- A CUDA box (Tasks 5-13 need it; Tasks 5/6/7/9/10 build the two HMR envs).
- Clone `git@github.com:yufu-wang/PromptHMR.git` into `~/code/PromptHMR/`.
- Create the `phmr_pt2.4` conda env per Task 5 (Python 3.11.9 + PyTorch 2.4
  + cu121, then `pip install -e .` from PromptHMR).
- Clone `git@github.com:facebookresearch/sam-body4d.git` into
  `~/code/sam-body4d/`.
- Create the `body4d` conda env per Task 6 (Python 3.12 + PyTorch 2.7.1
  + cu118, then `pip install -e .`).
- Hugging Face account that has accepted the gates for **SAM-3** and
  **SAM-3D-Body** (see Task 6 step 5).
- Then Tasks 7-14 can proceed: per-track JSON conversion, sidecar runners,
  comparison stage, multi-clip orchestration.

---

## 2026-04-17 — Lambda A100 provisioned

Operator has provisioned a **Lambda Cloud A100** for Tasks 5-14. Decisions
recorded in plan §10:
- All four stages (A, B, C1, C2, D) run on the A100; Stage A artifacts
  generated on Mac were only for Task 4 smoke test and will be regenerated
  on the box.
- HF gates: skip `facebook/sam3` (we use our SAM-2 masks); accept
  `facebook/sam-3d-body-dinov3` only.
- Checkpoints land in `~/checkpoints/` on local NVMe (~30 GB).

### Manual checklist BEFORE next session (operator does these on Lambda
or in browser; agent does not need SSH for any of this):

1. **Confirm A100 variant (40 GB vs 80 GB).** Tell agent next session.
   - 80 GB: every clip including loveTest works (with per-track batching).
   - 40 GB: adiTest / loveTest / BigTest force `--disable-completion`.
2. **Push the 8 commits to `https://github.com/arnavchokshi/SAM-HMR.git`**
   so the Lambda box can `git clone` instead of rsync. The agent did NOT
   push (per session constraint). From Mac:
       cd /Users/arnavchokshi/Desktop/yolo+bytetrack && git push -u origin main
3. **HuggingFace setup**:
   - Create / sign in to https://huggingface.co
   - Visit https://huggingface.co/facebook/sam-3d-body-dinov3 and click
     "Agree and access repository" (needs to read agreement; takes <1 min)
   - Create a Read token at https://huggingface.co/settings/tokens
   - Save the token somewhere — we will paste it into `huggingface-cli login`
     on the box during Task 6.
4. **Lambda SSH access**: confirm you can `ssh ubuntu@<ip>` (or
   `ssh lambda@<ip>`, depends on image) from your Mac. Lambda emails the
   IP and SSH key when the instance finishes spinning up.
5. **Source video access**: agent will `rsync` the 8 source clips
   (~few hundred MB total) from `/Users/arnavchokshi/Desktop/{adiTest,
   2pplTest, ...}/` to the box once SSH works. Nothing for operator to do.
6. **GitHub auth on the box** (only matters if the SAM-HMR repo stays
   private): either make the repo public for now, or generate a deploy
   key and add it to the repo. Public is simpler for this session.

When the IP is ready, paste it in chat and the agent will:
- ssh in, sanity-check `nvidia-smi` + driver version
- `git clone` SAM-HMR (containing our threed/ scaffolding + plan)
- `git clone` PromptHMR + sam-body4d under `~/code/`
- Build the two conda envs (Tasks 5 & 6)
- Re-run Stage A on adiTest from the box (~10 s on A100)
- Continue Tasks 7-14

---

## 2026-04-18 — Box-side preflight (Lambda A100, IP 150.136.209.7)

New session. Operator handed off Tasks 5–14 with explicit pre-flight checklist
and gate-stop policy (handoff §5, §9). Box-side preflight outcomes:

### Pre-flight findings

- **GPU:** `NVIDIA A100-SXM4-40GB`, NOT 80 GB as hand-off §3 stated.
  Driver 580.126.09, CUDA-13.0 compat (works for both cu121 and cu118
  PyTorch wheels). Box was idle (0 MiB VRAM, no other GPU processes).
- **Compute:** 30 vCPU, 216 GiB RAM, 484 GB free on `/`.
- **Other agent / user:** `~/dance_bench/` (5.9 GB — `uv .venv`, `repo/`,
  `clips/{adiTest,gymTest,BigTest}/gt`). Confined my work to `~/code/`,
  `~/checkpoints/`, `~/work/`, `~/miniforge3/`. No conflicts expected.
- **Conda:** none installed at session start. Installed Miniforge3
  (`conda 26.1.1`, `mamba 2.5.0`) into `~/miniforge3/`. `conda init bash`
  added the activate hook to `~/.bashrc`.
- **tmux:** session `arnav-3d` started for long-running installs.
- **GitHub auth:** SAM-HMR repo is public. Box can `git ls-remote` without
  credentials. No deploy key needed.

### Operator decisions (recorded from session-start AskQuestion)

1. **Push:** operator authorised the agent to `git push -u origin main` from
   the Mac (overrides the previous "operator pushes" rule). Done — origin
   head now `f0c921a`.
2. **HF token:** gate accepted for `facebook/sam-3d-body-dinov3` (operator
   confirmed "accepted" status in Hugging Face settings). Read token issued
   and pasted into chat for use on the box (will be consumed via
   `huggingface-cli login --token …`; never written to a tracked file).
   Operator should rotate the token when the project ends.
3. **SMPL credentials:** operator has both SMPL-X and SMPL accounts but
   prefers we reuse cached body models. Spotlight (`mdfind`) located a
   complete tree at:
   - `/Users/arnavchokshi/Desktop/sway_pose_mvp/PromptHMR/data/body_models/`
     (smpl/{NEUTRAL,MALE,FEMALE}.pkl real files; smplx/ symlink to
     `~/Desktop/sway_pose_mvp/models/smplx_body/models/smplx/`; plus
     `J_regressor_h36m.npy`, `smpl_mean_params.npz`, `smplx2smpl.pkl`,
     `smplx2smpl_joints.npy`).
   - `~/Desktop/sway_PHMR old Checkpoint/sync_body_models_to_lambda.sh`
     already exists (a prior sync script using `rsync -L` to deref symlinks).
   We will rsync this tree to the box and **skip** `bash scripts/fetch_smplx.sh`
   in plan Task 5 step 3 — saves ~30 min and avoids the interactive license
   prompts.
4. **Task numbering:** follow the plan's numbering verbatim; the hand-off
   §6 list is treated as milestone-gate descriptions, not a renumbering.
   Mapping table written into plan §11. Operator-report milestones A–E
   correspond to plan Tasks 5, 8, 11, 13(adiTest), 13(loveTest)+14 — see
   plan §11 mapping table.
5. **GPU 40 GB strategy:** try `completion.enable=true` first per clip; on
   OOM, fall back to `--disable-completion` for that clip and record the
   VRAM peak that triggered the fallback. Document per-clip outcomes in
   the §11 operator log table.

### Hand-off corrections caught during preflight

- Hand-off §6 / §9: SAM-Body4D upstream URL given as
  `facebookresearch/sam-body4d` (404). The plan §9 already correctly lists
  `gaomingqi/sam-body4d` (master branch, 299★, arXiv 2512.08406). We use
  the plan's URL. No plan edit needed.
- Hand-off §3: GPU stated as 80 GB but actually 40 GB — see "GPU strategy"
  decision above. Plan §10 question 1 already enumerates both variants and
  the 40 GB mitigation.

### Commits this session (preflight only — no code yet)

- `docs(plan): record 2026-04-18 box-side preflight + handoff mapping`
  (`runs/3d_compare/_agent_log.md` updates already committed in same set)

### Open questions for the user

- (none currently — all blockers resolved)

### Next actions

Proceed to plan Task 5 step 1 (clone PromptHMR into `~/code/PromptHMR/` on
the box) inside the `arnav-3d` tmux session.

---

## 2026-04-18 — Plan Task 5 complete (PromptHMR-Vid env + boxing smoke)

### What ran on the box

| Step | Wrapper script | tmux | Wall time | Outcome |
| --- | --- | --- | --- | --- |
| 5.1 clone | (manual `git clone`) | n/a | 12 s | `~/code/PromptHMR` HEAD `7d39d3f` |
| 5.2 install | `~/work/run_task5_install.sh` | `arnav-3d` | ~6 min | conda env `phmr_pt2.4` created (Python 3.11, torch 2.4.0+cu121, CUDA 12.1, all PromptHMR deps + DROID-SLAM + Detectron2) |
| 5.3a body models rsync | `scp` from Mac | n/a | ~6 min (3.37 GB) | `data/body_models/{smpl,smplx,*.npz,*.pkl}` populated from `/Users/arnavchokshi/Desktop/sway_pose_mvp/PromptHMR/data/body_models/`; `fetch_smplx.sh` skipped per operator decision |
| 5.3b BEDLAM2 ckpt | `wget` (inside `run_task5_fetch.sh`) | `arnav-3d` | ~25 s | `data/pretrain/phmr_vid/phmr_b1b2.ckpt` (472 MB) |
| 5.3c fetch_data.sh | `~/work/run_task5_fetch.sh` | `arnav-3d` | ~4.5 min | `phmr/`, `phmr_vid/`, `sam2_ckpts/`, `sam_vit_h_4b8939.pth` (2.4G), `vitpose-h-coco_25.pth` (2.4G), `camcalib_sa_biased_l2.ckpt` (288M), `droidcalib.pth` (16M), `examples/{boxing,boxing_short,dance_1,dance_2}.mp4` — 5.1 GB total |
| 5.3d slim SMPLX npz | `gdown` (manual, after demo first run) | n/a | 2 s | `data/body_models/smplx/SMPLX_neutral_array_f32_slim.npz` (69 MB, gdrive id `1v9Qy7…`) — needed by GLB export, not pulled by `fetch_data.sh`/`fetch_smplx.sh` automatically |
| 5.4 demo | `~/work/run_task5_demo.sh` then direct `convert_mcs_to_gltf` | `arnav-3d` | ~4 min wall (50 frames @ 25 fps, two boxers, `--static_camera`) | `results/boxing_short/{results.pkl 401K, world4d.mcs 44K, world4d.glb 66M, subject-{1,2}.smpl}` — gate A passed |

### Key engineering decisions

- **Did NOT swap to `phmr_b1b2.ckpt` for the smoke test.** Plan-correction
  commit `8c9232e` (`docs(plan): correct phmr_vid ckpt switch instructions`)
  documents that:
  1. The yaml has no `pretrained_ckpt` key — the path is hardcoded in
     `pipeline/phmr_vid.py:22`, which is what the upstream README's
     "modify the checkpoint path in this line" hyperlink points to.
  2. Running the smoke test against the bundled `prhmr_release_002.ckpt`
     first isolates env failures from ckpt-swap failures (one variable
     at a time). Swap will happen right before plan Task 7 wires
     PromptHMR-Vid into our sidecar runner.
- **Wrapper-script `set -e` bug.** `run_task5_demo.sh` uses
  `{ set -euo pipefail; …; touch "$DONE"; } >> "$LOG" 2>&1; echo "$?" > "$EXIT"`
  — when the inner block fails, `set -e` exits the whole script before
  `$DONE` is touched AND before the outer `echo "$?" > "$EXIT"` runs.
  Result: neither sentinel appeared after the slim-npz failure, even
  though most of the pipeline succeeded. Will switch to
  `set +e; cmd; ec=$?; …; echo "$ec" > "$EXIT"; if (( ec == 0 )); then touch "$DONE"; fi`
  for Task 6 onward.
- **Slim npz is a hidden dep.** `fetch_data.sh` does NOT download
  `SMPLX_neutral_array_f32_slim.npz`; only `fetch_smplx.sh` does (last
  line, `gdown 1v9Qy7…`). Because we skipped `fetch_smplx.sh` per
  operator decision (rsynced body models from Mac instead) we missed it.
  Documented as a footnote — for any future fresh box, EITHER run
  `fetch_smplx.sh` OR remember to `gdown` that file separately.

### VRAM peaks observed

| Stage | Peak VRAM (us) | GPU util sample |
| --- | --- | --- |
| Demo idle (model load + ViTPose + DETR weights) | 8.6 GB | 7 % |
| Pipeline (PRHMR-Vid + SLAM-static + post-opt) | 11.8 GB | 45 % (instantaneous post-completion) |

So the boxing-short run peaks at ~12 GB on a 50-frame clip with two
people. Comfortably inside the 40 GB envelope (other tenant uses ~3 GB).

### Hand-off correction (recap)

Hand-off §3 says PromptHMR install command is just
`bash scripts/install.sh --pt_version=2.4 --world-video=true`. Confirmed
correct on this box; produced a working env on the first try.

### Commits this session (Task 5)

- `8c9232e` — `docs(plan): correct phmr_vid ckpt switch instructions`
- (pending) `log: Task 5 complete (PromptHMR-Vid env + boxing_short smoke)`

### Operator report — milestone A

- Env `phmr_pt2.4` exists on the box and runs the boxing smoke demo
  end-to-end (results.pkl + world4d.mcs + world4d.glb).
- Disk used after Task 5: 37 GB / 484 GB free.
- Peak VRAM: 11.8 GB, comfortably inside 40 GB headroom.
- No code touched in our repo for Task 5 (it is purely a "stand up
  the upstream tool" milestone). Only the plan got a small correction.

### Next actions

Proceed to plan Task 6 (PromptHMR SAM-2 mask sidecar — `threed/sidecar_promthmr/build_masks.py`).
Plan Task 8 is the SAM-Body4D env install (came after Task 7 in the plan).

---

## 2026-04-18 — Plan Task 6 complete (PromptHMR SAM-2 mask sidecar)

### Source-of-truth deviation (recorded in plan §11 too)

Plan Task 6's original commit message put the sidecar inside the
PromptHMR clone (`PromptHMR/scripts/build_masks_from_bboxes.py`).
We instead implemented it inside our own repo at
`threed/sidecar_promthmr/build_masks.py`, for three reasons:

1. The PromptHMR clone is a vendored upstream repo on the box; keeping
   the sidecar in `SAM-HMR` lets it move with our git history and
   makes it `pytest`-coverable from the host repo.
2. The plan-level instruction "modify upstream PromptHMR" inflates the
   diff against `yufu-wang/main` and complicates future rebases.
3. The sidecar imports from `pipeline.detector.sam2_video_predictor`
   (PromptHMR's modified SAM-2 fork) by injecting `~/code/PromptHMR`
   into `sys.path` at runtime, so the *behaviour* is identical — only
   the on-disk location differs.

Plan-correction commit `75751ce` (`docs(plan): relocate SAM-2 mask
sidecar into our repo + use sam2_hiera_tiny.pt`) records this and
also bumps the default SAM-2 checkpoint to `sam2_hiera_tiny.pt` (the
smallest variant; PromptHMR's `fetch_data.sh` ships all three sizes).

### Implementation (TDD — red → green → refactor → green)

- **`threed/sidecar_promthmr/__init__.py`** — package marker.
- **`threed/sidecar_promthmr/build_masks.py`** — main entry. Public
  helpers (each with a unit test): `davis_palette`,
  `resolve_default_sam2_paths`, `valid_frames_set`,
  `assemble_palette_canvas`, `compute_union`, `inject_prompthmr_path`,
  `chdir_to_prompthmr`, `hydra_absolute_config_name`,
  `load_video_frames_bgr`. Internal: `_build_predictor`,
  `_propagate_with_predictor`, `_write_per_track_pngs`,
  `_write_palette_pngs`, `main`.
- **`tests/threed/test_sidecar_promthmr_build_masks.py`** — 22 GPU-free
  unit tests covering every helper. Box-side smoke test exercises the
  GPU path end-to-end on `adiTest`.

### Three runtime errors caught + fixed (each with regression test)

1. **`ModuleNotFoundError: No module named 'hmr4d'`.** PromptHMR's
   `pipeline.phmr_vid:7` does
   `sys.path.insert(0, 'pipeline/gvhmr')` — a *relative* path that
   only resolves when CWD is the PromptHMR root. Importing
   `pipeline.detector.sam2_video_predictor` (which transitively imports
   `pipeline.phmr_vid` via `pipeline/__init__.py`) blew up from any
   other CWD. Fix: `chdir_to_prompthmr` switches CWD to the PromptHMR
   root before the import, after the helpers have already absolutised
   every input path. Commit `9811b08`. Test:
   `TestChdirToPromptHmr::test_chdir_required_for_pipeline_gvhmr_relative_path`.
2. **`hydra.errors.MissingConfigException: Cannot find primary config
   'pipeline/sam2/sam2_hiera_t.yaml'`.** Hydra's `compose()` resolves
   non-absolute config names against the registered search paths,
   which include `pkg://sam2` (the upstream sam2 package). PromptHMR
   ships a *modified* `sam2_hiera_t.yaml` under
   `PromptHMR/pipeline/sam2/` (different `feat_sizes` for the custom
   `SAM2VideoPredictor` subclass) — Hydra silently picked the upstream
   one. Fix: build a Hydra-absolute config name (`'/' + abspath(cfg)`),
   the same trick PromptHMR's own `pipeline/tools.py:241` uses. Commit
   `aea669e`. Test:
   `TestHydraAbsoluteConfigName::test_double_slash_prefix_with_absolute_filesystem_path`.
3. **`TypeError: SAM2VideoPredictor.init_state() missing 1 required
   positional argument: 'video_frames'`.** PromptHMR's modified
   `init_state` (`pipeline/detector/sam2_video_predictor.py:78`)
   replaced upstream's `video_path` argument with a stacked
   `video_frames` numpy array (it does
   `video_frames.shape[1:3]`). The in-class `_load_img_as_tensor`
   accepts BGR uint8 numpy frames without channel-swap. Fix: add
   `load_video_frames_bgr(frames_dir)` which returns
   `(N, H, W, 3) uint8`. Commit `761cf27`. Tests: 4 cases under
   `TestLoadVideoFramesBgr`.

### Smoke test on `adiTest` intermediates

Wrapper `~/work/run_task6_smoke.sh` (uses inner exit-code capture —
fixes the `set -e` wrapper bug from Task 5). Run inside `arnav-3d`
tmux. Highlights from `~/work/logs/task6_smoke.log` (HEAD `761cf27`):

```
[build_masks] 188 frames @ 1280x720, 5 tids,
              sam2 ckpt=sam2_hiera_tiny.pt, cfg=pipeline/sam2/sam2_hiera_t.yaml
Loaded checkpoint sucessfully
[build_masks] wrote 940 per-tid PNGs, 188 palette PNGs,
              union shape=(188, 720, 1280) sum=10026161
build_masks exit=0
```

| Metric | Value |
| --- | --- |
| Wall time | 28 s |
| Peak VRAM (sam2 hiera_tiny) | <1 GB (under nvidia-smi sample resolution) |
| Per-tid PNGs | 940 (= 5 tids × 188 frames) |
| Palette PNGs (P-mode, DAVIS palette) | 188 (verified via PIL: `mode='P'`, indices `[0,1,2,3,4,5]`) |
| Union mask | `(188, 720, 1280) bool`, total True pixels 10 026 161 |

PIL spot-check confirmed the palette PNG is true 8-bit indexed mode
(OpenCV `imread` upgrades P-mode PNGs to RGB so `cv2.imread` reports
the resolved colour values, not the underlying tids — use PIL when
you need the indices).

### VRAM peaks observed during Task 6

| Stage | Peak VRAM | Notes |
| --- | --- | --- |
| sam2 hiera_tiny init + propagate (188 frames, 5 tids) | <1 GB | Reported as 0 MiB at end-of-job by `nvidia-smi --query-gpu=memory.used` (sampled too late); peak during prop is still well under 4 GB based on the model size + frame batch. |

### Commits this session (Task 6)

- `75751ce` — `docs(plan): relocate SAM-2 mask sidecar into our repo + use sam2_hiera_tiny.pt`
- `90765eb` — `[Plan Task 6] sidecar masks: scaffold + 15 unit tests (red → green)`
- `9811b08` — `[Plan Task 6] sidecar masks: chdir to PromptHMR root before SAM-2 import (fixes hmr4d ModuleNotFoundError)`
- `aea669e` — `[Plan Task 6] sidecar masks: pass Hydra-absolute config name (fixes MissingConfigException)`
- `761cf27` — `[Plan Task 6] sidecar masks: stack frames into video_frames numpy array`
- (pending) `log: Task 6 complete (PromptHMR SAM-2 mask sidecar)`

### Open questions for the user

- (none — all blockers resolved)

### Next actions

Proceed to plan Task 7 (PromptHMR-Vid sidecar runner —
`threed/sidecar_promthmr/run_promthmr_vid.py`). This is also where we
finally swap the hardcoded checkpoint path in
`PromptHMR/pipeline/phmr_vid.py:22` from the bundled
`prhmr_release_002.ckpt` to the BEDLAM2-trained
`phmr_b1b2.ckpt` (per plan-correction commit `8c9232e`).

---

## 2026-04-18 — Plan Task 7 complete (PromptHMR-Vid sidecar runner)

### Source-of-truth deviation

Plan Task 7 step 1 originally placed the runner inside the
PromptHMR clone (`our_pipeline/run_phmr.py`); we instead put it in
our repo at `threed/sidecar_promthmr/run_promthmr_vid.py` for the
same reasons documented in Task 6 (single git history, `pytest`
coverage from the host repo, no upstream fork patches).

### Implementation (TDD — red → green → 4 runtime fixes → green)

- **`threed/sidecar_promthmr/run_promthmr_vid.py`** — Stage C1 runner.
  GPU-free helpers: `intermediates_layout_ok` (checks every artifact
  in one shot for clear errors), `load_per_track_masks` (mutates
  `tracks` to add `masks`/`track_id`/`detected` per the contract
  PromptHMR's `Pipeline` expects), `sorted_tid_list` (stable python-int
  sort tolerant of joblib's `np.int64` keys), `joints_world_padded`
  (builds the `(n_frames, n_dancers, 22, 3)` NaN-padded comparison
  artifact). GPU helpers: `_load_frames_rgb`, `_extract_smplx_body_joints_world`,
  `_write_smpl_and_world4d`, plus the integration `main`.
- **`tests/threed/test_sidecar_promthmr_run_promthmr_vid.py`** — 15
  GPU-free pytest cases covering every helper above.
- The end-to-end PromptHMR-Vid call (DROID-SLAM + ViTPose +
  PRHMR-Vid + post-opt) is exercised by the box-side smoke test
  (CUDA-only).

### Operator-side checkpoint swap (one-time)

Per plan-correction commit `8c9232e`, this is where we finally swap
`PromptHMR/pipeline/phmr_vid.py:23` from the bundled
`prhmr_release_002.ckpt` to the BEDLAM2-trained `phmr_b1b2.ckpt`.
The wrapper script (`~/work/run_task7_smoke.sh`) does the swap with
an idempotent `sed` (no-ops if already swapped) and prints the
post-swap line for the audit trail. The swap is ALSO the only
on-disk modification to `~/code/PromptHMR/`; everything else lives
in our repo.

### Four runtime errors caught + fixed during the smoke test

1. **`UnboundLocalError: cannot access local variable 'imgsize' where
   it is not associated with a value`** in
   `pipeline.spec.cam_calib.run_spec_calib`. Upstream's branch:
   `if isinstance(images, np.ndarray): … elif isinstance(images[0], str): …`
   leaves `imgsize` undefined for `list[np.ndarray]`. Fix:
   `_load_frames_rgb` now returns a stacked 4D `np.ndarray` (matches
   PromptHMR's `load_video_frames`), so the first branch matches.
   Commit `86413c3`.
2. **`RuntimeError: Sizes of tensors must match except in dimension 1.
   Expected size 188 but got size 1 for tensor number 2`** in
   `smplx.SMPLX.forward`. The forward unconditionally cats
   `global_orient + body_pose + jaw + leye + reye + l_hand + r_hand`
   along `dim=1`, so leaving any of them unset reuses
   `self.jaw_pose` etc. (initialised at batch=1) and trips the
   dim-0 mismatch. Fix: pass jaw/eye as zero tensors with batch=B
   and route left+right hand from `pose[:,75:120] / pose[:,120:165]`
   — exactly mirrors PromptHMR's own call in `pipeline/world.py:104`.
   Commits `d33af1f` (axis-angle direct slicing) → `7f18d7c`
   (jaw/eye/hand additions).
3. *(self-doc)* The intermediate `axis_angle_to_matrix(...)` +
   `pose2rot=True` combination from the original plan stub corrupts
   `body_pose` shape because SMPL-X's `pose2rot=True` reshapes (-1,
   J, 3) on the rot-matrix tensor. Removed; we pass axis-angle
   directly and let SMPL-X's default `pose2rot=True` do the
   conversion.
4. *(architectural)* Added `--reuse-results` flag (commit `59c9cb1`)
   that skips SLAM/ViTPose/PRHMR-Vid/world if `<output-dir>/results.pkl`
   already exists — saves ~5 min per iteration when debugging the
   export tail.

### Smoke test on `adiTest` intermediates (fresh end-to-end run)

Wrapper `~/work/run_task7_smoke.sh` (inner exit-code capture +
1 Hz background VRAM sampler). Successful end-to-end run inside
`arnav-3d` tmux at HEAD `59c9cb1`:

```
[run_promthmr_vid] wrote .../results.pkl, joints_world.npy (188, 5, 22, 3),
                   world4d.mcs/glb, 5 subject-*.smpl files
run_promthmr_vid exit=0
  vram peak (MiB): 10911
  joints_world.npy shape=(188, 5, 22, 3) dtype=float32 nan_frac=0.000
```

| Metric | Value |
| --- | --- |
| Wall time (start → exit) | 1 min 48 s on adiTest (188 frames, 5 dancers, static-camera) |
| Peak VRAM | 10 911 MiB (~10.7 GB) — well inside 40 GB envelope |
| `results.pkl` | 1.0 GB |
| `joints_world.npy` shape | `(188, 5, 22, 3)` float32, NaN fraction 0.0 |
| `world4d.mcs` | 348 KB (188 frames, 5 bodies, fps 30) |
| `world4d.glb` | 74 MB (auto-converted via `convert_mcs_to_gltf`) |
| `subject-{1..5}.smpl` | 5 files, ~50 KB each |

Joint sanity (using `pipeline.smplx` forward on `smplx_world`):

| Axis | Min | Max | Comment |
| --- | --- | --- | --- |
| X (lateral) | -3.37 m | +2.95 m | 5 dancers spanning ~6 m laterally — sensible |
| Y (height) | +0.009 m | +1.76 m | feet near floor, hands up to head — sensible |
| Z (depth)  | -11.74 m | -5.34 m | static-camera world is camera-centric so all dancers in front (Z negative); will be re-aligned in Stage D if needed |

### VRAM peaks observed during Task 7

| Stage | Peak VRAM | Notes |
| --- | --- | --- |
| ViTPose-h + PRHMR-Vid (1× clip, 188 frames, 5 dancers) | 10.7 GB | Static camera (no DROID-SLAM); use_spec_calib=True; post_opt enabled |

(Sampled at 1 Hz via background `nvidia-smi --query-gpu=memory.used`.)

### Hand-off correction (recap)

The plan stub set `cfg.tracker = "external"` to flag "we provided
external tracks". That value is NOT recognised by `pipeline.hps_estimation`,
which gates `mask_prompt = (cfg.tracker == "sam2")`. We set
`cfg.tracker = "sam2"` so PromptHMR-Vid actually consumes our
SAM-2 masks. Documented inline in the runner's main().

### Commits this session (Task 7)

- `c8f341f` — `[Plan Task 7] PromptHMR-Vid sidecar runner: scaffold + 15 unit tests`
- `86413c3` — `[Plan Task 7] runner: stack frames into 4D ndarray (fixes spec_calib UnboundLocalError)`
- `d33af1f` — `[Plan Task 7] runner: pass axis-angle directly to SMPL-X (fixes shape mismatch)`
- `7f18d7c` — `[Plan Task 7] runner: pass jaw/eye/hand axis-angle to SMPL-X (mirrors world.py:104)`
- `59c9cb1` — `[Plan Task 7] runner: add --reuse-results to skip the heavy inference path`
- (pending) `log: Task 7 complete (PromptHMR-Vid sidecar runner)`

### Open questions for the user

- (none — all blockers resolved)

### Next actions

Proceed to plan Task 8 (`body4d` conda env + clone `gaomingqi/sam-body4d`
+ Gradio demo on a single image — milestone gate B per the operator
mapping in plan §11). HF gate for `facebook/sam-3d-body-dinov3` was
already accepted during preflight; the runner-side SAM-3 monkey-patch
lands later in plan Task 9.

---

## 2026-04-18 — Plan Task 8 complete (body4d env + sam-body4d clone + headless smoke)

### Source-of-truth deviations

1. **Step 4 (Gradio UI verification → headless wrapper-path smoke).** The
   plan calls for `python app.py` and clicking "Generate Masks → Generate
   4D" in the Gradio UI. Over an SSH-only Lambda box (no X11, no port
   forward) interactive verification is impractical. We instead validate
   the **production wrapper path** described in plan §3.4 (which monkey-
   patches `build_sam3_from_config → (None, None)` and feeds external
   palette masks). This is a stricter test of what we will actually run
   in Tasks 9–14, not just a smoke of the bundled demo.
2. **Install fix not in the plan: `setuptools<80`.** detectron2 0.6 imports
   `pkg_resources` at module load, which was removed from setuptools 80+.
   The body4d env ships setuptools 82 by default; we downgrade to 79.0.1
   so detectron2's `model_zoo.LazyConfig.load` works. This needs to be
   appended to plan Task 8 step 2 for future operators.
3. **Step 3 partial completion: SAM3 ckpt blocked.** The user's HF gate
   for `facebook/sam3` was still PENDING when we ran Task 8 (per
   `https://huggingface.co/settings/gated-repos`). All other 7 ckpts
   downloaded successfully (20 GB). Setup script reports `[BLOCKED] SAM3`
   and continues. SAM3 ckpt is **not needed** by our wrapper — it is only
   used by the bundled `scripts/offline_app.py` for first-frame mask
   propagation, which we replace entirely with our DeepOcSort + SAM-2
   pipeline (plan §3.4). We will re-run `setup.py` to grab `sam3.pt`
   once the gate flips, just for completeness, but it is not on the
   critical path.

### Box state after Task 8

```
~/code/sam-body4d/                  (HEAD 21af102 "update uni-cam-int", depth=1)
├── configs/body4d.yaml              (generated by setup.py with ckpt_root absolute path)
├── scripts/{offline_app.py,setup.py}
└── ...

~/checkpoints/body4d/                (20 GB)
├── depth_anything_v2_vitl.pth                                       1.34 GB
├── diffusion-vas-amodal-segmentation/                               7.6 GB (.complete marker)
├── diffusion-vas-content-completion/                                7.6 GB (.complete marker)
├── moge-2-vitl-normal/model.pt                                      ~1.5 GB
├── sam-3d-body-dinov3/{model.ckpt,assets/mhr_model.pt,...}          2.8 GB
└── sam3/                                                            (empty — gate pending)

~/miniforge3/envs/body4d/            (Python 3.12.13)
├── torch                            2.7.1+cu118
├── detectron2                       0.6 (commit a1ce2f9, --no-deps)
├── sam3                             0.1.0 (editable from models/sam3)
├── sam-body4d                       0.1.0 (editable from root)
├── setuptools                       79.0.1 (downgraded from 82.0.1)
├── numpy                            2.2.6 (note: sam3 wants 1.26 but works)
└── ... (full freeze in ~/work/logs/task8_install.log)
```

Two known dependency conflicts logged but not blocking:

- `sam3 requires numpy==1.26 but you have 2.2.6` — sam-body4d's `pip install
  -e .` pulled the newer numpy. The SAM-3D-Body forward path works fine on
  numpy 2.2.6 in our smoke. We watch for runtime breakage in Tasks 9-10
  and re-pin only if needed.
- `detectron2 requires black, hydra-core, tensorboard, iopath<0.1.10` —
  intentional `--no-deps` per upstream sam-body4d README; only the
  vitdet inference path is used and it does not need these.

### Headless smoke test (`~/work/task8_smoke_wrapper.py`)

Validates the full **wrapper path** (Tasks 9–10's hot path):

| Step | Result |
| --- | --- |
| Monkey-patch `build_sam3_from_config → (None, None)` | OK |
| Extract first 8 frames from `assets/examples/example1.mp4` (854×480) | OK |
| Override config: `completion.enable=False`, `sam_3d_body.batch_size=8` | OK |
| `OfflineApp.__init__` (loads SAM-3D-Body + MoGe-2 FOV + vitdet detector) | 72.4 s, peak VRAM 7.7 GB (one-time vitdet download 2.77 GB → `~/.torch/iopath_cache`) |
| `vitdet.process_one_image(frame0)` | bbox `(507,53)-(600,221)` (single dancer in scene) |
| Write 8 palette PNGs (full bbox = obj_id 1, DAVIS palette) | OK |
| `app.on_4d_generation()` | 12.0 s, peak VRAM 9.44 GB |
| Outputs | 8 PLY meshes (`mesh_4d_individual/1/`), 8 focal JSONs, 8 rendered overlay JPGs, 1 `4d_*.mp4` |

What is intentionally NOT validated:

- **SAM-3 propagation** — gate pending; not on our critical path
  (production wrapper bypasses SAM-3 entirely via the same monkey-patch
  as the smoke).
- **Diffusion-VAS occlusion completion** — disabled here for fast smoke;
  will validate during Task 10's `body4d` runner with
  `completion.enable=True` (the resources.md table shows ~26 m runtime +
  53 GB peak on H800 for 5 dancers, so we'll need batch=32 + per-track
  batches of ≤5 IDs on our 40 GB A100).

### Resource envelope reminder (from `assets/doc/resources.md`)

40 GB A100 fits the SAM-Body4D 4D step *only with* careful settings:

| Scenario | OK on 40 GB? | Knobs |
| --- | --- | --- |
| 1 dancer, completion off, batch=64 | YES (14.5 GB) | default |
| 5 dancers, completion off, batch=64 | NO (40.9 GB) | drop batch to 32 |
| 5 dancers, completion on, batch=32 | YES (35.2 GB) | dance default |
| 5 dancers, completion on, batch=64 | NO (53.3 GB) | OOM |
| 6 dancers, completion on, batch=32 | borderline (34.8 GB) | watch closely |
| 15 dancers (loveTest) | NO at any batch | per-track batches of ≤5 IDs (linear in dancer count, not batch) |

Plan Task 10 will encode these as automatic per-clip overrides in the
runner.

### Commits this session (Task 8)

- (no source code in our repo changed for Task 8 — this is a box-only
  install + clone + smoke; logs and plan row only)
- (pending) `log: Task 8 complete (body4d env + sam-body4d clone + headless wrapper-path smoke)`

### Open questions for the user

- (none — SAM3 gate pending is not blocking; the wrapper does not need it)

### Next actions

Proceed to plan Task 9 (`SAM-Body4D wrapper` — sidecar runner that
materialises external palette masks from our DeepOcSort + SAM-2 outputs,
monkey-patches SAM-3, calls `on_4d_generation`). This is the production
sibling of `threed/sidecar_promthmr/run_promthmr_vid.py` and lives in
our repo at `threed/sidecar_body4d/run_body4d.py`. Per the same
git-hygiene rationale as Tasks 6 and 7, we keep the runner inside this
repo (single test suite, single git history) rather than inside
`~/code/sam-body4d/our_pipeline/` as the original plan stub suggested.

---

## 2026-04-18 — Plan Task 9 complete (SAM-Body4D wrapper helpers)

### Scope split between Task 9 and Task 10

Plan §11 splits the SAM-Body4D sidecar across two tasks:

- **Task 9 (this section): wrapper helpers.** GPU-free, no torch, no
  cv2, no SAM-Body4D imports. Pure functions that can run in any
  conda env (including the host repo's `pose-tracking` env). Unit-tested
  on the laptop, no box round-trip needed.
- **Task 10 (next section): runner script.** Combines the helpers with
  GPU-side `OfflineApp.on_4d_generation` + COCO-17 joint extraction
  from the emitted PLYs + per-frame focal JSONs.

This split keeps the unit-testable surface large and predictable while
keeping the 9-GB GPU loader out of CI.

### What we wrote (`threed/sidecar_body4d/wrapper.py`)

| Helper | Purpose |
| --- | --- |
| `monkeypatch_sam3(module)` | Replaces `build_sam3_from_config → (None, None)`. Idempotent (sentinel `_sam3_patched_by_threed` on the patched module). No-clobber on other module attrs. |
| `intermediates_layout_ok(interm)` | Returns `(ok, errs)` for `frames_full/`, `masks_palette/`, `tracks.pkl`. Hard-fails on frame≠mask count to surface Task 6 truncation early. |
| `sorted_tid_list(tracks)` | Stable python-int sort of joblib's `np.int64` keys. Mirrors the PromptHMR helper of the same name (duplicated rather than shared so the test boundary stays per-sidecar). |
| `link_artifacts_into_workdir(out, frames_full, masks_palette)` | Symlinks both into `OUTPUT_DIR/{images,masks}/`. Symlinks vs copies saves ~hundreds of MB per clip; SAM-Body4D's `glob` follows symlinks. Idempotent — wipes existing children before relinking. |
| `workdir_layout_ok(out)` | Post-link sanity check that `images/` and `masks/` are 1:1 by basename. Used as a pre-flight inside the runner (cheap, surfaces issues before the slow GPU init). |
| `iter_palette_obj_ids(track_ids)` | Validates each tid ∈ [1..255] (palette PNG range). Guards against an upstream tracker bug emitting tid=0 (background) or tid>255 — neither has happened in our clips, but loveTest with 15 dancers is well within the limit. |

### Test coverage (`tests/threed/test_sidecar_body4d_wrapper.py`)

28 unit tests across 6 classes. Highlights:

- `TestMonkeypatchSam3`: 4 tests covering replacement, idempotency,
  sentinel marking, and no-clobber of `OfflineApp` / `RUNTIME` attrs.
- `TestIntermediatesLayoutOk`: 7 tests covering each missing
  artifact, count-mismatch detection, and error aggregation (so the
  operator sees all problems in one shot, not just the first).
- `TestSortedTidList`: 4 tests including `np.int64` keys and mixed
  int/numpy keys (the actual joblib output shape).
- `TestLinkArtifactsIntoWorkdir`: 4 tests including idempotent re-link,
  count-mismatch raise, and parent-dir auto-creation.
- `TestWorkdirLayoutOk`: 3 tests including basename mismatch detection
  (the failure mode if Task 6 wrote a partial mask set).
- `TestIterPaletteObjIds`: 6 tests including dedup, numpy inputs, the
  `[1..255]` guard at both ends, and the boundary case of tid=255.

Full suite: 70/70 pass under 0.6 s (no GPU, no body4d env required).

### Commits this session (Task 9)

- (pending) `feat(threed/sidecar_body4d): wrapper helpers + 28 unit tests (plan Task 9)`
- (pending) `log: Task 9 complete (SAM-Body4D wrapper helpers)`

### Open questions for the user

- (none — wrapper is pure-python and validated locally; Task 10's GPU
  runner is the next blocker check)

### Next actions

Proceed to plan Task 10 (`SAM-Body4D sidecar runner` — Stage C2). The
runner imports the wrapper helpers, applies the SAM-3 monkey-patch on
the box, instantiates `OfflineApp`, runs `on_4d_generation`, then
extracts COCO-17 joints from the per-frame PLY meshes via the
SAM-3D-Body MHR regressor (or a vertex-to-joint fallback if the MHR
exposes one). Outputs land in `runs/3d_compare/<clip>/body4d/` mirroring
the `prompthmr/` layout from Task 7, with `joints_cam.npy` ready for
Stage D's pairwise comparison.
