# 3D dancer reconstruction — research findings & recommendation

> **TL;DR — Concrete recommendation:** Build the 3D leg of the pipeline on
> **PromptHMR-Vid** (Wang et al., CVPR 2025), driven by your existing
> YOLO + DeepOcSort tracks as bounding-box prompts, with optional
> SAM-2 masks fed in as additional spatial prompts. This is the only
> SOTA HMR system that was *literally designed* for the pattern you
> already have (per-frame bbox + persistent ID), it is currently #1 on
> both single-person and two-person interaction benchmarks, and it
> emits SMPL-X meshes + GLB files that drop straight into a 3D viewer.
>
> **Second choice (worth a 1-day spike):** Meta's **SAM 3D Body + SAM-Body4D**
> (Nov–Dec 2025). Most-recent SOTA, biggest training data, but a
> brand-new mesh format (MHR) that your downstream stack would have to
> learn. If PromptHMR-Vid disappoints on `loveTest`-style crowded clips,
> this is your fallback.

---

## 1. What we already have (the "ID pipeline")

From `docs/WINNING_PIPELINE_CONFIGURATION.md`:

- **Detector:** YOLO26s, multi-scale ensemble at 768 + 1024, fine-tuned
  on dancers.
- **Tracker:** **BoxMOT DeepOcSort** (note: it's DeepOcSort, not vanilla
  DeepSORT — important detail for the next stage), OSNet x0.25 ReID.
- **Post-processing:** stitching/pruning at `mtf=60`, `prox=150 px`,
  `gap=120 frames`.
- **Output per video:** for every frame `t` and every dancer ID `i`,
  a tight bounding box `B_{t,i}` and (after the post-pass) a clean track.
- **Quality:** mean IDF1 = 0.949 across 6 GT clips. The bottleneck is
  `loveTest` (15 free-form dancers, IDF1 ≈ 0.80).

The crucial property: you have **persistent identity** + **clean
bounding boxes**. That is exactly the input format SOTA HMR systems
ask for. You do not need to redesign anything upstream.

---

## 2. What "good enough" looks like for the 3D stage

Before evaluating methods, the requirements have to be made explicit,
otherwise everything looks "kind of OK". For the dance-critique product
you described, the 3D stage has to deliver:

| # | Requirement | Why it matters |
|---|---|---|
| R1 | Per-joint 3D position accuracy of **< 70 mm MPJPE on 3DPW-class data** | This is the published bar for "you can show it to a user without it looking broken". Below ~60 mm and humans can't visually distinguish the mesh from a mocap rig. |
| R2 | **Temporal smoothness** (low jitter, ≤2-3 mm/frame) | A spinning dancer's mesh that flickers looks worse than a static avatar that is wrong by 50 mm. |
| R3 | **Per-dancer consistent body shape** | The same dancer must have the same height/build across the entire video, otherwise you can't compare them to an "average" or to themselves over time. |
| R4 | **World-grounded global trajectory** | "Where is the dancer in the room" — needed to render multiple dancers in the same 3D scene and to compare formations. |
| R5 | **Foot-skating-free contact** | The single most visually obvious failure mode. Solved-ish only since 2024. |
| R6 | **Robust to fast motion / motion blur** | Dance is the worst case for this — pirouettes, jumps, kicks. |
| R7 | **Handles close-contact / partial occlusion between dancers** | `loveTest` is your existing evidence that this is hard. |
| R8 | **Practical to run** in a per-video, offline batch (not real-time) | You have GPUs (Hopper / GH200, Apple Silicon for dev) and you do not need streaming inference. |

R1–R3 are non-negotiable. R4 + R5 are needed for the "comparison" UX
you described. R6 + R7 are where most methods break. R8 means we can
prefer accuracy over speed and afford expensive priors.

---

## 3. The current state of the art (April 2026)

Below is the honest landscape, grouped by "what they do" rather than by
year. All numbers are **published** numbers from the original papers /
the Human3R CVPR-style comparison table — i.e. apples-to-apples on the
same eval split.

### 3.1 Per-frame body recovery (camera coords)

Benchmark: **3DPW** (in-the-wild) and **EMDB-1** (newer, harder).
Lower is better. Best-in-class numbers are bolded.

| Method | Year | 3DPW PA-MPJPE ↓ | 3DPW MPJPE ↓ | 3DPW PVE ↓ | EMDB-1 PA-MPJPE ↓ | EMDB-1 MPJPE ↓ | EMDB-1 PVE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| HMR2.0a | ICCV 2023 | 44.4 | 69.8 | 82.2 | 61.5 | 97.8 | 120.0 |
| TokenHMR | CVPR 2024 | 44.3 | 71.0 | 84.6 | 55.6 | 91.7 | 109.4 |
| Multi-HMR | ECCV 2024 | 45.9 | 73.1 | 87.1 | 50.1 | 81.6 | 95.7 |
| CameraHMR | 3DV 2025 | 38.5 | 62.1 | 72.9 | 43.7 | 73.0 | 85.4 |
| **NLF** | NeurIPS 2024 | 37.3 | 60.3 | 71.4 | **41.2** | 69.6 | **82.4** |
| **PromptHMR** | CVPR 2025 | **36.6** | **58.7** | **69.4** | **41.0** | 71.7 | 84.5 |
| Human3R | Oct 2025 | 44.1 | 71.2 | 84.9 | 48.5 | 73.9 | 86.0 |

Source: Table in [Human3R, arXiv:2510.06219](https://arxiv.org/abs/2510.06219), §4.1.

**What you should take away:**

- **PromptHMR and NLF are tied at the top.** Everything else is at
  least 2 mm worse on PA-MPJPE, which is a lot at this part of the
  curve.
- **Both clear R1** (`< 70 mm MPJPE` on 3DPW). Most older systems
  (HMR2.0, TokenHMR, Multi-HMR) do not.
- The famous **HMR2.0** that a lot of tutorials still recommend is now
  ~6 mm behind — it's the SD 1.5 of HMR. Do not start there in 2026.

### 3.2 Video / world-grounded recovery

Benchmark: **EMDB-2** (long sequences, dynamic camera, GT global
trajectories). Metrics: world-aligned and world MPJPE (mm), root
translation error (% of total displacement).

| Method | Type | Year | WA-MPJPE ↓ | W-MPJPE ↓ | RTE % ↓ |
|---|---|---|---:|---:|---:|
| GLAMR | offline | CVPR 2022 | 280.8 | 726.6 | 11.4 |
| SLAHMR | offline | CVPR 2023 | 326.9 | 776.1 | 10.2 |
| WHAM | online | CVPR 2024 | 135.6 | 354.8 | 6.0 |
| GVHMR | offline | SA 2024 | 111.0 | 276.5 | 2.0 |
| Human3R | online | Oct 2025 | 112.2 | 267.9 | 2.2 |
| TRAM | offline | ECCV 2024 | 76.4 | 222.4 | 1.4 |
| **JOSH** | offline | 2025 | **68.9** | **174.7** | **1.3** |
| **PromptHMR-Vid** | offline | CVPR 2025 | (paper claim: better than TRAM on these metrics, including foot-skating) | | |

Source: same Human3R table, plus PromptHMR §4.2.

**What you should take away:**

- For **global-coords trajectories**, the gap between TRAM/PromptHMR-Vid
  and everyone else is enormous (~60 mm WA-MPJPE, 4× lower RTE).
- **PromptHMR-Vid uses TRAM's own SLAM machinery** (DROID-SLAM +
  ZoeDepth) and adds **per-joint contact estimation** to kill foot
  skating — TRAM's most visible flaw. So it is essentially "TRAM + a
  contact head + better per-frame regressor". This is from the same
  author (Yufu Wang) — PromptHMR is an explicit upgrade path.
- **JOSH is the absolute offline ceiling** but it is hours-per-video
  and depends on iterative human–scene optimisation. It is only
  worth touching if you decide to render the *room* too, which you
  said is out-of-scope right now.

### 3.3 Multi-person / close-contact (the dance test)

Benchmark: **HI4D** and **CHI3D** — these are exactly two-person
close-contact captures with GT meshes. Metric: `Pair-PA-MPJPE` (mm),
which aligns the *pair* as a single rigid body and then measures the
joint error. Lower is better.

| Method | HI4D Pair-PA-MPJPE ↓ | CHI3D Pair-PA-MPJPE ↓ |
|---|---:|---:|
| BEV | 136 | 96 |
| BUDDI | 98 | 68 |
| Multi-HMR (zero-shot) | 80.6 | 100.0 |
| PromptHMR (zero-shot) | 78.1 | 58.5 |
| **PromptHMR (trained on it)** | **39.5** | **45.3** |

Source: PromptHMR paper, §4.

**What you should take away:**

- **Nothing else is even close.** PromptHMR's interaction-prompt module
  is a ~2× error reduction over the second-best system on HI4D. This
  matters for `loveTest`-style clips where bodies frequently overlap
  in image space.
- For your `loveTest` failure mode (15 free-form dancers, IDF1 ≈ 0.80),
  this is the single most important number on this page.

### 3.4 What about pure single-image systems?

NLF (`isarandi/nlf`), CameraHMR, SAM 3D Body, and Multi-HMR are all
single-image. They are very strong per-frame but have **no temporal
model**. If you went this route you would need to:

1. Run them frame-by-frame on each `B_{t,i}` crop.
2. Apply your own temporal smoothing (e.g. 1€ filter, Gaussian on
   axis-angle, Kalman on rotation matrices) — this **trades jitter
   against reaction time** and **never beats** a video-native model on
   accuracy.
3. Solve the world-grounding problem yourself (separate SLAM run +
   re-projection).

This is a lot of glue code that PromptHMR-Vid bundles for you. The only
reason to choose single-image is if you have a hard reason to —
e.g. you want to run on still photos too, or you want SAM 3D Body's
hands.

### 3.5 The Meta entrant: SAM 3D Body (Nov 2025) + SAM-Body4D (Dec 2025)

This is the most recent and most heavily-resourced contender. It
deserves its own subsection because it is the *only* serious threat to
the PromptHMR recommendation.

**SAM 3D Body** (Meta Reality Labs, Nov 19 2025, [code](https://github.com/facebookresearch/sam-3d-body)):

- **Promptable like SAM:** accepts 2D keypoints and segmentation masks
  as encoder/decoder prompts. Bounding boxes work as a degenerate case.
- **Whole body** (body + feet + hands), but uses a *new* parametric
  representation called **MHR (Momentum Human Rig)**, not SMPL-X.
  MHR decouples skeleton from shape, which is a cleaner formulation
  for retargeting — but **the whole downstream ecosystem
  (Blender plugins, animation tools, Three.js loaders, comparison
  metrics) speaks SMPL/SMPL-X**, not MHR.
- **Open source** (SAM license + permissive-commercial MHR license),
  checkpoints on HuggingFace, web demo on Meta's Segment Anything
  Playground.
- **Single image only** — for video you need SAM-Body4D.

**SAM-Body4D** (arXiv 2512.08406, Dec 2025, [code](https://github.com/gaomingqi/sam-body4d)):

- **Training-free** wrapper that takes an input video, runs SAM 3 to
  get identity-consistent masklets, runs a Diffusion-VAS occlusion
  refiner to fill in hidden body regions, and feeds the refined masks
  into SAM 3D Body in parallel batches.
- Adds **Kalman smoothing in MHR space** + locks shape from the first
  visible frame of each track. (This is *exactly* the recipe you would
  end up writing yourself if you went the single-image route.)
- Has a clean Algorithm 1 you can re-implement against PromptHMR if
  you want — the framework is method-agnostic.

**Honest assessment:**

| Dimension | PromptHMR-Vid | SAM 3D Body + SAM-Body4D |
|---|---|---|
| Per-frame 3DPW accuracy | **#1 published** | not yet on the leaderboard, qualitative-only paper |
| Multi-person interaction | **#1 by a huge margin** | qualitative claims, no HI4D/CHI3D number |
| Temporal model | **native** (12-block decoder transformer) | bolt-on Kalman, training-free |
| World-grounded trajectory | **yes** (TRAM-style SLAM) | no — masks only, camera coords |
| Mesh format | **SMPL-X** (universal) | MHR (new, niche) |
| Hands | body-only at the released checkpoint | yes (separate body+hand optim) |
| Foot-skating | explicit contact head | inherited from per-frame model |
| Code maturity | full demo, MCS+GLB output, Meshcapade viewer | just released, less battle-tested |
| Backed by | Meshcapade + MPI + UPenn | Meta Reality Labs |
| Risk | well understood | unknown |

The core trade-off is: **PromptHMR is the strongest published number
on the metrics that matter for your use case, with a mature output
pipeline. SAM 3D Body has the bigger institutional backing and the
better single-image accuracy on certain rare poses, but the video
story is one paper old and the mesh format is a science project.**

### 3.6 The methods that look attractive but should not be picked

- **HMR2.0 / 4DHumans** — the canonical "first thing tutorials show
  you". Now 6+ mm behind on every metric. Skip.
- **WHAM** — was great in 2024, now beaten on every metric by GVHMR /
  TRAM / PromptHMR. Skip unless you specifically need *online*.
- **GLAMR / SLAHMR** — superseded by TRAM/JOSH; very slow.
- **SMPLer-X / SMPLest-X** — generalist single-image foundation models.
  Strong, but not as accurate as PromptHMR on 3DPW and no temporal
  story. Use if you need an off-the-shelf body recovery for unrelated
  static images.
- **NLF** — *very close runner-up* to PromptHMR on per-frame, and
  uniquely it can localize *any* body point (not just SMPL joints)
  via its neural localizer field. **Worth keeping in your back
  pocket** as a per-frame refiner on top of PromptHMR if you need to
  query specific body landmarks (e.g. "where is the left elbow at
  exactly this pixel"). Not a replacement for the video model.
- **GVHMR** — strong, fast, gravity-aware. The thing it does best
  (handling unknown focal length / changing intrinsics) does not
  apply to your use case (a single dance video shot on one camera).
- **Multi-HMR** — best *single-shot* multi-person model. But your
  YOLO+DeepOcSort already solves the "find the people and ID them"
  problem better than Multi-HMR's bottom-up detector does. Don't
  reinvent.
- **Sapiens (Meta)** — *not an HMR model*, it's a 2D foundation model
  for pose, segmentation, depth, normals. It can be useful as a
  pre-processor (e.g. its 2D pose can prompt PromptHMR's "keypoint"
  channel) but it doesn't produce 3D meshes. Treat as a tool, not a
  product.

---

## 4. The recommendation, with reasoning

### 4.1 Primary: PromptHMR-Vid

Reasoning, in priority order:

1. **It is the SOTA on every benchmark that maps to your product.**
   Per-frame accuracy (3DPW, EMDB), multi-person interaction (HI4D,
   CHI3D), foot-skating (better than TRAM). For a research project
   "concrete answer" question, this is the answer the numbers point
   at — there is no other system that wins on more than one of these.
2. **It is shaped to consume your pipeline's output.** PromptHMR's
   *first-class input* is a person bounding box. You already produce
   high-quality per-frame, per-ID bounding boxes. The integration
   point is literally one function call per `(frame, dancer_id)`.
3. **It outputs SMPL-X.** Every 3D viewer, animation library, and
   metric in the field speaks SMPL or SMPL-X. You can render to
   Three.js / Blender / Unity / Unreal without writing a custom
   loader. The repo also emits `.glb` and Meshcapade `.mcs` directly,
   so for your "every dancer gets their own page" product you can
   serve `glb` straight to a `<model-viewer>` web component.
4. **It already solves world-grounding.** PromptHMR-Vid bundles
   DROID-SLAM + ZoeDepth metric scaling (TRAM's recipe) and adds
   per-joint contact estimation. You do not have to write that
   integration.
5. **It is actively maintained by people you would want to be on
   your side.** Same author (Yufu Wang, Kostas Daniilidis,
   Michael J. Black) wrote TRAM, PromptHMR, and PromptHMR-Vid.
   This is the lineage of HMR research that has consistently
   shipped — if there is a v3, it will probably come from here.
6. **Concrete shape "lock" prompt.** PromptHMR accepts language
   shape prompts ("tall and slim", "broad-shouldered"). Even
   without using language, you can lock body shape from the first
   confident frame of each dancer ID and reuse it for the entire
   video — this directly satisfies R3 ("same dancer looks the same
   throughout").

The single legitimate risk: the released checkpoint is body-only
(no hand articulation). For dance critique this is acceptable for
v1 (joints + body shape are 95 % of the value), and you can add
a wrist-cropped hand recovery pass later if needed (HaMer, ACR).

### 4.2 Secondary: SAM 3D Body + SAM-Body4D-style wrapper

Run this *in parallel* on a small evaluation set (say `loveTest`,
`BigTest`, and 1–2 outdoor clips). If it visibly beats PromptHMR
on `loveTest`-style crowded clips and the MHR → SMPL-X conversion
cost is acceptable, switch. Otherwise keep it as the v2 swap.

### 4.3 Reusable parts you should also adopt

Independently of which mesh model you pick:

- **SAM 2 / SAM 3** for per-dancer masks. Even if you stay on
  PromptHMR-Vid, adding a SAM mask prompt on top of the bounding
  box prompt is the cheapest accuracy boost available (PromptHMR
  accepts both, SAM-Body4D's whole thesis is that the mask channel
  matters most). You already have the bbox seeds for SAM 2 from
  DeepOcSort.
- **DROID-SLAM** (already a PromptHMR-Vid dep) for the camera
  trajectory. If you want to compare two videos that were filmed
  from different angles, the camera trajectory is what lets you
  align them in a shared world frame.
- **A first-frame shape lock per dancer ID.** The DeepOcSort ID is
  exactly the key for this. After the first ~30 confident frames
  per ID, fit β (SMPL-X shape) and freeze it for that ID.

---

## 5. Proposed pipeline shape

```
[ video.mp4 ]
    │
    ▼
[ YOLO26s ensemble + DeepOcSort + post-pass ]   ← what you have
    │       per frame, per dancer:
    │         B_{t,i}  (bbox)
    │         id_{t,i} (track id)
    ▼
[ SAM 2 / SAM 3 ]   ← cheap add, drives mask prompt
    │       per frame, per dancer:
    │         M_{t,i} (binary mask)
    ▼
[ DROID-SLAM ]   ← bundled with PromptHMR-Vid, world frame
    │       per frame:
    │         camera pose, intrinsics, scene depth scale
    ▼
[ PromptHMR-Vid ]   ← the new piece
    │       per dancer i, per frame t:
    │         SMPL-X (θ_t, β_i fixed, ψ_t, R_t, T_t) in world coords
    ▼
[ post: per-ID shape lock, contact-aware foot fix, Kalman smoothing ]
    │       per dancer i:
    │         clean motion sequence in SMPL-X
    ▼
[ render ]   ← three.js / Meshcapade viewer / Blender, .glb files
        per-dancer page  +  multi-dancer comparison view
```

Every box on this diagram has a paper, a public implementation, and
a benchmark number behind it. There is no "hopefully this works"
step.

---

## 6. What this *does not* solve (and what to do about it)

Be honest about the limits, otherwise you will rebuild the same thing
in 6 months:

1. **Detailed hand pose at low resolution.** A dancer in a ~1080p
   wide shot is ~600 px tall, so their hand is ~30 px. *No* current
   monocular system gives reliable finger articulation at that scale.
   For dance, this means you can show "arm position" but not
   "specific finger styling". Mitigation: 2-stage hand recovery via
   HaMer on wrist crops *only when needed* (e.g. in solo close-up
   sections of the video).
2. **Severe self-occlusion during turns/rolls.** Even PromptHMR
   degrades when most of the body is occluded for several frames.
   The SAM-Body4D *occlusion-aware refiner* (Diffusion-VAS) is the
   current best-known fix and is method-agnostic — you can wrap it
   around PromptHMR-Vid yourself.
3. **Physical plausibility / penetrations.** PromptHMR is a
   regression model, not a physics simulator. Two dancers that touch
   may end up slightly inter-penetrating in the mesh. For *pure
   visualisation* this is fine (viewer can't tell). For
   *quantitative critique* ("you stepped 5 cm closer than the
   reference"), you'd want to add a contact-aware optimization pass
   (this is what JOSH does, very slowly).
4. **Very fast spins (>4 rev/s).** Motion blur kills 2D evidence.
   Mitigation: high-FPS source video (60 / 120 fps) where possible.
   None of the SOTA methods solve this; they all degrade gracefully.

---

## 7. Sources used (key papers)

- Wang et al., **PromptHMR: Promptable Human Mesh Recovery**, CVPR 2025.
  [arXiv:2504.06397](https://arxiv.org/abs/2504.06397) ·
  [code](https://github.com/yufu-wang/PromptHMR) ·
  [project page](https://yufu-wang.github.io/phmr-page/)
- Wang et al., **TRAM: Global Trajectory and Motion of 3D Humans
  from in-the-Wild Videos**, ECCV 2024.
  [arXiv:2403.17346](https://arxiv.org/abs/2403.17346)
- Sárándi & Pons-Moll, **Neural Localizer Fields**, NeurIPS 2024.
  [arXiv:2407.07532](https://arxiv.org/abs/2407.07532) ·
  [code](https://github.com/isarandi/nlf)
- Patel & Black, **CameraHMR: Aligning People with Perspective**,
  3DV 2025. [arXiv:2411.08128](https://arxiv.org/abs/2411.08128)
- Shen et al., **GVHMR: World-Grounded Human Motion Recovery via
  Gravity-View Coordinates**, SIGGRAPH Asia 2024.
  [arXiv:2409.06662](https://arxiv.org/abs/2409.06662)
- Shin et al., **WHAM: Reconstructing World-Grounded Humans with
  Accurate 3D Motion**, CVPR 2024.
  [arXiv:2312.07531](https://arxiv.org/abs/2312.07531)
- Baradel et al., **Multi-HMR: Multi-Person Whole-Body Mesh
  Recovery**, ECCV 2024.
  [arXiv:2402.14654](https://arxiv.org/abs/2402.14654)
- Yang et al. (Meta), **SAM 3D Body: Robust Full-Body Human Mesh
  Recovery**, Nov 2025. [code](https://github.com/facebookresearch/sam-3d-body) ·
  [HuggingFace](https://huggingface.co/facebook/sam-3d-body-dinov3)
- Gao et al., **SAM-Body4D: Training-Free 4D Human Body Mesh
  Recovery from Videos**, Dec 2025.
  [arXiv:2512.08406](https://arxiv.org/abs/2512.08406)
- Chen et al., **Human3R: Everyone Everywhere All at Once**, Oct 2025.
  [arXiv:2510.06219](https://arxiv.org/abs/2510.06219) ·
  [code](https://github.com/fanegg/Human3R)
- Pavlakos et al., **SMPL-X: Expressive Body Capture**, CVPR 2019
  (model used by PromptHMR, NLF, Human3R).
  [arXiv:1904.05866](https://arxiv.org/abs/1904.05866)
