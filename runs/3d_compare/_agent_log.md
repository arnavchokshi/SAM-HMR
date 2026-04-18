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

Proceed to plan Task 6 (clone SAM-Body4D, create `body4d` conda env,
run bundled Gradio demo on a single image to confirm SAM-3-Body works
on the box).
