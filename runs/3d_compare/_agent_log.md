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
