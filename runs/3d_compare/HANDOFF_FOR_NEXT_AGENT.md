# Handoff prompt — dual 3D HMR pipeline (Tasks 5–13 done, Task 14 partial)

> Copy everything **between the two `---` lines** into the new agent's
> first user message. Tweak the "What I want you to do" section to
> match what you actually want next.

---

# Hand-off — Dual 3D HMR pipeline (Lambda A100 phase, picking up after Task 14 partial)

## Context

The prior agent finished plan **Tasks 5–13** (milestones A → D) on
`adiTest` end-to-end and shipped a **partial Task 14** (operator report
on adiTest only; loveTest + 5 other clips deferred pending GPU-spend
authorisation). Everything is committed to `main` of
`https://github.com/arnavchokshi/SAM-HMR.git` and pushed up to commit
**`1734c84`**. You are authorised to `git push -u origin main`.

You **must** read these three files before doing anything (they are the
ground truth for state, decisions, and what's left):

1. `docs/3D_DUAL_PIPELINE_PLAN.md` — the canonical plan. Pay special
   attention to §11 ("Operator log", lines ~2660–2720) where every
   completed task has a row with commit refs, metrics, and
   deviations.
2. `runs/3d_compare/_agent_log.md` — the full per-task agent log
   (decisions, source-of-truth deviations, box state captures,
   full smoke-test results, open questions per task).
3. `runs/3d_compare/REPORT.md` — the operator report. §5 has the
   exact step-by-step procedure to extend to a new clip; §6 has
   five open followups (Procrustes-aligned MPJPE, world-frame
   foot-skating, PHMR mesh overlays, 2D reprojection vs ViTPose,
   optional HTML report).

After those three, skim:
- `scripts/run_3d_compare.py` — the orchestrator. Pure
  command-builders + `plan_pipeline()` skip-flag composer are
  unit-tested in `tests/threed/test_orchestrator.py`.
- `threed/sidecar_body4d/wrapper.py` and `run_body4d.py` — Stage C2.
- `threed/sidecar_promthmr/{build_masks.py,run_promthmr_vid.py,project_joints.py}` — Stage B + C1.
- `threed/compare/{joints,metrics,render,run_compare}.py` — Stage D.

## Box access

- Lambda A100-SXM4-**40 GB** at `ubuntu@150.136.209.7`
  (SSH key `~/.ssh/pose-tracking.pem`)
- Repo clone: `~/work/SAM-HMR` (do `git pull --ff-only origin main`
  before any GPU run to pick up local commits)
- Conda envs: `body4d` (py3.12.13, torch 2.7.1+cu118),
  `phmr_pt2.4` (py3.10, torch 2.4.0+cu121)
- Per-clip artifact root: `~/work/3d_compare/<clip>/{intermediates,prompthmr,sam_body4d,comparison}/`
- Long-running jobs go in tmux session `arnav-3d`. Logs live in
  `~/work/logs/`.
- Upstream clones: `~/code/PromptHMR @ 7d39d3f`, `~/code/sam-body4d @ 21af102`
- Checkpoints: `~/checkpoints/body4d/` (~20 GB). SAM-3 (`sam3.pt`) is
  intentionally NOT downloaded — we monkey-patch it out
  (`threed.sidecar_body4d.wrapper.monkeypatch_sam3`); the
  `facebook/sam-3d-body-dinov3` HF gate is the only one that's been
  granted. Don't try to re-enable SAM-3 unless the user asks.
- VRAM budget per body4d run: ~12 GB on adiTest (5 dancers,
  `--disable-completion --batch-size 32`); leave room for completion-on
  + bigger dancer counts.

## What's currently on disk

`adiTest` is fully validated end-to-end through the orchestrator:

- `~/work/3d_compare/adiTest/sam_body4d/joints_world.npy` — `(188, 5, 70, 3)` MHR70 cam-frame
- `~/work/3d_compare/adiTest/prompthmr/joints_coco17_cam.npy` — `(188, 5, 17, 3)` COCO-17 cam-frame
- `~/work/3d_compare/adiTest/comparison/{metrics.json, side_by_side.mp4}`
- 940 PLYs / 940 focal JSONs / 940 joint .npy / 188 rendered overlays
- `run_summary.json` with timings + VRAM

**No other clip has artifacts yet** — they need tracking caches first
(see "Required prerequisites per new clip" below).

## What I want you to do

Pick the scope I confirm in chat, but the realistic options are:

1. **Single clip extension (loveTest only):** ~30-60 min Mac (tracking)
   + ~50-60 min A100 (orchestrator with completion ON, batch=16).
2. **Full multi-clip batch (loveTest + easyTest + gymTest + 2pplTest +
   BigTest):** ~3-5 h GPU + ~2-4 h Mac total. Stop after each clip
   and surface summary metrics so I can decide whether to continue.
3. **Open followups before more clips:** land #1 (Procrustes MPJPE)
   first because the current 9.21 m flat MPJPE is uninformative; it's
   a ~1-2 h coding task with TDD.
4. **HTML report:** convert `runs/3d_compare/REPORT.md` to a
   self-contained HTML with embedded `side_by_side.mp4` per clip.

Whichever you pick, follow the **rules in the next section verbatim**.

## Rules (non-negotiable)

1. **TDD on host first.** For any new module / wrapper / metric:
   write the unit test before the implementation, get it to fail for
   the right reason, then implement. Run `python -m pytest tests/threed/ -q`
   locally before pushing. Current count: 161 passed + 3 skipped.
2. **GPU-only smoke runs go in tmux on the box** (window names like
   `task14-loveTest`). Tee logs to `~/work/logs/<task>.log`. Always
   poll with `tail -n 25` then `tmux capture-pane -t arnav-3d:<window> -p | tail -n 25`.
3. **Update both logs after every task:**
   - One row in `docs/3D_DUAL_PIPELINE_PLAN.md` §11 ("Operator log").
   - One section in `runs/3d_compare/_agent_log.md` with: source-of-truth
     deviations, box state, smoke results (timings + VRAM + output
     shapes), architectural decisions, open questions, commit refs,
     "Next actions".
4. **Commit messages follow the existing pattern**
   (`feat(threed/...): ...`, `fix(...): ...`, `log: ...`, `docs: ...`).
   See `git log --oneline -20` for the cadence. Bug fixes get their
   own commits with the regression test included. Use HEREDOC for
   multi-line messages.
5. **Deviations from the plan are OK** but record them explicitly in
   `_agent_log.md` AND the plan §11 row. Do NOT silently "improve"
   things.
6. **Don't re-enable SAM-3.** Don't fork sam-body4d or PromptHMR.
   Sidecar everything via `threed/sidecar_*/` and monkey-patch where
   needed (the `monkeypatch_sam3` and `monkeypatch_save_mesh_results`
   patterns are the references).
7. **Don't run unauthorised GPU spend.** If a job will take >1 h on
   the A100, surface a brief "this will cost ~$X over Y hours, OK to
   proceed?" before launching.
8. You can `git push -u origin main` (this was authorised by the user).

## Required prerequisites per new clip

The orchestrator's Stage A reads from a **YOLO+DeepOcSort tracking
cache** (a `.pkl`). Only `adiTest`'s exists in this repo
(`runs/winner_stack_demo/_cache/adiTest/imgsz768_conf0.310_iou0.700_boxmot_deepocsort_457d921a09e4_max188.pkl`).

To stage a new clip, e.g. `loveTest`:

```bash
# (A) On the Mac (uses MPS) or on the box (uses CUDA), produce the cache:
python scripts/run_winner_stack_demo.py \
    --clip loveTest \
    --video /Users/arnavchokshi/Desktop/loveTest/IMG_9265.mov

# (B) Sync video + cache to the box:
ssh -i ~/.ssh/pose-tracking.pem ubuntu@150.136.209.7 'mkdir -p /home/ubuntu/work/videos /home/ubuntu/work/cache/loveTest'
scp -i ~/.ssh/pose-tracking.pem \
    /Users/arnavchokshi/Desktop/loveTest/IMG_9265.mov \
    ubuntu@150.136.209.7:/home/ubuntu/work/videos/
scp -i ~/.ssh/pose-tracking.pem \
    runs/winner_stack_demo/_cache/loveTest/*.pkl \
    ubuntu@150.136.209.7:/home/ubuntu/work/cache/loveTest/

# (C) Drive the orchestrator on the box (in tmux):
ssh -i ~/.ssh/pose-tracking.pem ubuntu@150.136.209.7
tmux new-window -t arnav-3d -n task14-loveTest
source ~/miniforge3/etc/profile.d/conda.sh && conda activate body4d
cd ~/work/SAM-HMR && git pull --ff-only origin main
python scripts/run_3d_compare.py \
    --clip loveTest \
    --output-root /home/ubuntu/work/3d_compare \
    --video /home/ubuntu/work/videos/IMG_9265.mov \
    --cache-dir /home/ubuntu/work/cache/loveTest \
    --batch-size 16 \
    --fps 30 \
  > ~/work/logs/task14_loveTest.log 2>&1
```

(For loveTest's ~15 dancers, drop `--batch-size` to 16 or 8 to stay
under 40 GB VRAM with completion ON. See plan §3.5 mitigations.)

## Candidate clip videos (on user's Mac)

| Clip | Path | Approx dancers |
|---|---|---:|
| `adiTest` (DONE) | `/Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov` | 5 |
| `loveTest` | `/Users/arnavchokshi/Desktop/loveTest/IMG_9265.mov` | ~15 |
| `easyTest` | `/Users/arnavchokshi/Desktop/easyTest/IMG_2082.mov` | ~3 |
| `gymTest` | `/Users/arnavchokshi/Desktop/gymTest/IMG_8309.mov` | ~3 |
| `2pplTest` | `/Users/arnavchokshi/Desktop/2pplTest/2pplTest.mov` | 2 |
| `BigTest` | `/Users/arnavchokshi/Desktop/BigTest/BigTest.mov` | ~6 |

## Open followups (REPORT.md §6 — pick before / after / interleaved with multi-clip)

1. **Procrustes-aligned MPJPE.** Add `align_procrustes(joints_a,
   joints_b, per_dancer=True)` to `threed/compare/metrics.py`; report
   MPJPE before AND after. Target after-alignment MPJPE: <50 cm for
   COCO body joints. (TDD: 5-8 unit tests for identity / translation /
   rotation / per-dancer / NaN propagation.)
2. **World-frame foot-skating.** Compute foot-skating using
   un-projected world-frame joints from PHMR (read `joints_world.npy`
   directly; don't re-project) with a per-clip-calibrated ground
   plane. Currently PHMR's cam-frame foot-skating is `[0,0,0,0,0]`
   because the camera is mounted above the floor.
3. **PHMR mesh overlays.** Render SMPL-X meshes from
   `prompthmr/results.pkl` into `prompthmr/rendered_frames/<frame:08d>.jpg`
   and pass to `--prompthmr-frames-dir` of the orchestrator's render
   step. Use `pyrender` or borrow PromptHMR's bundled renderer.
4. **2D reprojection vs ViTPose.** Both pipelines should reproject
   their 3D joints into image space using their per-frame intrinsics
   and compare against the bundled ViTPose 17-keypoint output that
   PromptHMR-Vid emits in `results.pkl["vitpose"]`.
5. **HTML report.** Self-contained `runs/3d_compare/report.html` with
   embedded `<video>` tags per clip. Keep `REPORT.md` as the
   markdown source of truth.

## What's in `runs/3d_compare/` right now

```
runs/3d_compare/
├── _agent_log.md       — full per-task agent log (READ THIS FIRST)
└── REPORT.md           — operator report, ships in repo
```

(All per-clip artifacts live under `~/work/3d_compare/<clip>/` on the
box, NOT in this repo — they're too large. The downloaded videos for
adiTest are at `/Users/arnavchokshi/Desktop/3d_compare_outputs/adiTest/`
on the user's Mac.)

## How to verify you're not regressing anything

```bash
cd ~/work/SAM-HMR  # or local clone
python -m pytest tests/threed/ -q
# expected: 161 passed, 3 skipped, 1 warning  (≤2 s)
```

If that drops below 161, fix it BEFORE moving on.

## First task

Tell the user the very first thing you'll do is read the three "must
read" files (plan §11, `_agent_log.md`, `REPORT.md`) and propose a
concrete first concrete action with rough wall + cost estimate. Then
wait for confirmation before launching anything that costs GPU time.

---
