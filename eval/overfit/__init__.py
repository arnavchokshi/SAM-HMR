"""Overfitting / generalization audit suite.

The Optuna sweep optimises directly against scores on the same 4-5 clips
it then "ranks" by, so a high ``held_out_mean`` does NOT guarantee the
winning config will work on a fresh, never-seen dance video. This package
provides four orthogonal probes that quantify that risk **without**
requiring more annotated clips:

* ``sensitivity``      Per-knob postprocess sensitivity. A robust config
                       has small score deltas under \xb110-50% knob
                       perturbations; an overfit config sits on a knife-
                       edge where any nudge collapses one or more clips.
* ``per_clip_oracle``  For each clip, search for the postprocess config
                       that maximises *that* clip's score in isolation.
                       The "regret" between the per-clip oracles and a
                       single global config is a direct measure of how
                       much different videos require different params.
* ``no_gt_score``      Track-stability scorer that does NOT use GT, so
                       it can be applied to any new video to flag bad
                       runs at inference time. Captures count flicker,
                       jitter, persistence, and aspect plausibility.
* ``synth_perturb``    Apply ffmpeg-based input perturbations (downscale,
                       brightness, hflip, frame-decimation) and re-score
                       a config on the perturbed clips. A config that
                       only works on the exact pixel layout it was tuned
                       on will collapse here.

All four can run on the YOLO+tracker outputs that ``eval/eval_counts.py``
already cached on disk for the 6 user clips, so most of the audit
finishes in seconds, with no GPU and no risk of disturbing the running
Lambda sweep.

For pass/fail criteria, audit results to date, and how to evaluate a
new candidate config, see ``docs/OVERFITTING.md``.

Quick start::

    # Audit the active hand-tuned config (sensitivity + transferability + no-GT)
    python -m eval.overfit.audit --no-synth-perturb \\
        --output runs/overfit_analysis/audit_handtuned

    # Audit a sweep candidate (point at any sweep_optuna top_configs/<id>.json)
    python -m eval.overfit.audit \\
        --post-cfg-json runs/sweep_gh200/top_configs/<trial_id>.json \\
        --output runs/overfit_analysis/audit_<name>

    # Trust score on a brand-new, unannotated video (no GT required)
    python -m eval.overfit.no_gt_score --video /path/to/new_clip.mp4
"""
