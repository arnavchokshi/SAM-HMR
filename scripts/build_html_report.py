"""Followup #5 — build a self-contained HTML operator report.

Walks ``--root`` (default ``runs/3d_compare/``) for per-clip
sub-directories that contain ``comparison/metrics.json``, summarises
the headline metrics into a single dashboard table, and generates a
per-clip section with the ``side_by_side.mp4`` linked relative to the
HTML so the whole bundle (HTML + per-clip videos) can be served
from a static webserver or opened locally.

Why this is a separate script (and not a Stage E in the orchestrator):

- it's a *cross-clip* synthesis — the orchestrator runs per-clip,
- it has no GPU dependency and is cheap enough to re-run on every
  metrics update (think: "did Procrustes change adiTest?"),
- its outputs are read by humans, so iterating on the layout
  shouldn't trigger any pipeline re-run.

Pure host module — numpy + stdlib. The HTML uses inline CSS only
(no external CDNs) so it works air-gapped and over file://.
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Discovery + summarisation (testable + framework-agnostic)
# ---------------------------------------------------------------------------


def discover_clips(root: Path) -> List[Path]:
    """Return per-clip directories under ``root`` that have a ``comparison/metrics.json``.

    Sorted alphabetically so the HTML output order is deterministic
    across runs (otherwise diff-based PR review of the report churns).
    """
    out: List[Path] = []
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "comparison" / "metrics.json").is_file():
            out.append(p)
    return out


def _safe_nanmean(x) -> Optional[float]:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return None
    import warnings
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = float(np.nanmean(a))
    if not np.isfinite(v):
        return None
    return v


def summarize_clip(clip_dir: Path) -> Dict:
    """Read a single clip's metrics + reproj JSON into a dict-of-scalars row.

    Missing optional metrics (world FS or reproj — clips processed before
    Followups #2 / #4 won't have them) are returned as ``None`` so the
    HTML renderer can show a placeholder cell instead of crashing.
    """
    metrics_path = clip_dir / "comparison" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    summary: Dict = {
        "name": clip_dir.name,
        "n_frames": int(metrics.get("n_frames_compared", 0)),
        "n_dancers": int(metrics.get("n_dancers_compared", 0)),
        "mean_jitter_phmr_m": _safe_nanmean(metrics.get("per_joint_jitter_phmr_m_per_frame")),
        "mean_jitter_body4d_m": _safe_nanmean(metrics.get("per_joint_jitter_body4d_m_per_frame")),
        "mean_mpjpe_raw_m": _safe_nanmean(metrics.get("per_joint_mpjpe_m")),
        "mean_mpjpe_pa_m": _safe_nanmean(metrics.get("per_joint_mpjpe_pa_m")),
        "mean_foot_skating_phmr_cam_m": _safe_nanmean(metrics.get("foot_skating_phmr_m_per_frame")),
        "mean_foot_skating_body4d_cam_m": _safe_nanmean(metrics.get("foot_skating_body4d_m_per_frame")),
        "mean_foot_skating_phmr_world_m": _safe_nanmean(metrics.get("foot_skating_phmr_world_m_per_frame")),
        "side_by_side_path": "comparison/side_by_side.mp4",
        "metrics_path": "comparison/metrics.json",
    }
    reproj_path = clip_dir / "comparison" / "reproj_metrics.json"
    if reproj_path.is_file():
        reproj = json.loads(reproj_path.read_text())
        phmr_px = reproj.get("mean_mpjpe_phmr_vs_vitpose_px", None)
        b4d_px = reproj.get("mean_mpjpe_body4d_vs_vitpose_px", None)
        summary["mean_reproj_mpjpe_phmr_px"] = (
            float(phmr_px) if phmr_px is not None and np.isfinite(phmr_px) else None
        )
        summary["mean_reproj_mpjpe_body4d_px"] = (
            float(b4d_px) if b4d_px is not None and np.isfinite(b4d_px) else None
        )
        summary["reproj_n_low_conf"] = int(reproj.get("n_low_confidence_keypoints", 0))
        summary["reproj_path"] = "comparison/reproj_metrics.json"
    else:
        summary["mean_reproj_mpjpe_phmr_px"] = None
        summary["mean_reproj_mpjpe_body4d_px"] = None
        summary["reproj_n_low_conf"] = None
        summary["reproj_path"] = None
    return summary


def aggregate_clip_metrics(root: Path) -> List[Dict]:
    """Return ``summarize_clip`` rows for every discoverable clip."""
    return [summarize_clip(c) for c in discover_clips(root)]


# ---------------------------------------------------------------------------
# HTML rendering (inline CSS, no JS, no external deps)
# ---------------------------------------------------------------------------


_HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
  :root {{
    --fg: #1a1a1a;
    --muted: #5b6770;
    --bg: #fafafa;
    --card: #ffffff;
    --accent: #2b6cb0;
    --good: #2f855a;
    --warn: #b7791f;
    --bad: #c53030;
    --rule: #e2e8f0;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    line-height: 1.5;
  }}
  header {{
    background: linear-gradient(180deg, #1a365d 0%, #2c5282 100%);
    color: #fff;
    padding: 32px 40px;
  }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; font-weight: 700; }}
  header .subtitle {{ font-size: 14px; opacity: 0.9; }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--rule);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  h2 {{ margin-top: 0; font-size: 22px; }}
  h3 {{ margin-top: 0; font-size: 18px; color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }}
  th, td {{
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--rule);
  }}
  th {{
    background: #f7fafc;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    font-size: 12px;
    letter-spacing: 0.04em;
  }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: #f7fafc; }}
  .clip {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 24px; align-items: start; }}
  .clip video {{ width: 100%; border-radius: 6px; background: #000; }}
  .clip .meta dl {{ margin: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 6px 16px; }}
  .clip .meta dt {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .clip .meta dd {{ margin: 0 0 6px; font-size: 16px; font-variant-numeric: tabular-nums; font-weight: 500; }}
  .pill {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  .pill.good {{ background: #c6f6d5; color: var(--good); }}
  .pill.warn {{ background: #feebc8; color: var(--warn); }}
  .pill.bad {{ background: #fed7d7; color: var(--bad); }}
  .placeholder {{
    background: #f7fafc;
    border: 1px dashed var(--rule);
    padding: 40px;
    text-align: center;
    color: var(--muted);
    border-radius: 6px;
  }}
  footer {{ padding: 24px 40px; color: var(--muted); font-size: 12px; text-align: center; }}
  @media (max-width: 800px) {{ .clip {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
"""


_HTML_FOOT = """
<footer>Generated by <code>scripts/build_html_report.py</code> &middot; dual 3D HMR pipeline (PromptHMR-Vid &times; SAM-Body4D)</footer>
</body>
</html>
"""


def _fmt(value: Optional[float], fmt: str = "{:.3f}") -> str:
    if value is None:
        return "<span style='color:#a0aec0'>n/a</span>"
    if isinstance(value, float) and not np.isfinite(value):
        return "<span style='color:#a0aec0'>n/a</span>"
    return fmt.format(value)


def _pa_class(pa_m: Optional[float]) -> str:
    if pa_m is None:
        return ""
    if pa_m < 0.30:
        return "good"
    if pa_m < 0.70:
        return "warn"
    return "bad"


def _winner_cell(phmr: Optional[float], b4d: Optional[float]) -> str:
    """Format the head-to-head reproj cell with the winner highlighted."""
    if phmr is None or b4d is None:
        return f"<td class='num'>{_fmt(phmr, '{:.2f}')}<br><span style='color:#a0aec0'>vs {_fmt(b4d, '{:.2f}')}</span></td>"
    if phmr < b4d * 0.95:
        return (
            f"<td class='num'><b style='color:#2f855a'>{phmr:.2f} PHMR</b><br>"
            f"<span style='color:#5b6770'>{b4d:.2f} Body4D</span></td>"
        )
    if b4d < phmr * 0.95:
        return (
            f"<td class='num'><b style='color:#2f855a'>{b4d:.2f} Body4D</b><br>"
            f"<span style='color:#5b6770'>{phmr:.2f} PHMR</span></td>"
        )
    return (
        f"<td class='num'>{phmr:.2f} PHMR<br>{b4d:.2f} Body4D <span style='color:#a0aec0'>(tied)</span></td>"
    )


def _build_summary_table(rows: List[Dict]) -> str:
    if not rows:
        return "<p>No clips found under root.</p>"
    head = """<table>
<thead><tr>
  <th>Clip</th><th>Frames</th><th>Dancers</th>
  <th>Raw MPJPE (m)</th>
  <th>PA-MPJPE (m)</th>
  <th>Jitter PHMR (m/f)</th>
  <th>Jitter Body4D (m/f)</th>
  <th>Foot-skating PHMR world (m/f)</th>
  <th>Foot-skating Body4D cam (m/f)</th>
  <th>Reproj-vs-ViTPose (px) &mdash; PHMR vs Body4D</th>
</tr></thead><tbody>"""
    body = []
    for r in rows:
        pa_pill = _pa_class(r.get("mean_mpjpe_pa_m"))
        pa_cell = (
            f"<td class='num'>{_fmt(r.get('mean_mpjpe_pa_m'))} "
            f"<span class='pill {pa_pill}'>"
            f"{'ok' if pa_pill == 'good' else 'high' if pa_pill == 'bad' else 'mid' if pa_pill == 'warn' else ''}"
            f"</span></td>"
        ) if pa_pill else f"<td class='num'>{_fmt(r.get('mean_mpjpe_pa_m'))}</td>"
        body.append(
            "<tr>"
            f"<td><a href='#clip-{html_lib.escape(r['name'])}'>{html_lib.escape(r['name'])}</a></td>"
            f"<td class='num'>{r['n_frames']}</td>"
            f"<td class='num'>{r['n_dancers']}</td>"
            f"<td class='num'>{_fmt(r.get('mean_mpjpe_raw_m'))}</td>"
            f"{pa_cell}"
            f"<td class='num'>{_fmt(r.get('mean_jitter_phmr_m'))}</td>"
            f"<td class='num'>{_fmt(r.get('mean_jitter_body4d_m'))}</td>"
            f"<td class='num'>{_fmt(r.get('mean_foot_skating_phmr_world_m'), '{:.4f}')}</td>"
            f"<td class='num'>{_fmt(r.get('mean_foot_skating_body4d_cam_m'), '{:.4f}')}</td>"
            f"{_winner_cell(r.get('mean_reproj_mpjpe_phmr_px'), r.get('mean_reproj_mpjpe_body4d_px'))}"
            "</tr>"
        )
    return head + "\n".join(body) + "</tbody></table>"


def _build_clip_section(r: Dict, root: Path) -> str:
    name = r["name"]
    safe = html_lib.escape(name)
    video_rel = f"{name}/{r['side_by_side_path']}"
    video_abs = root / video_rel
    video_html = (
        f"<video controls preload=\"metadata\" poster=\"\">"
        f"<source src=\"{video_rel}\" type=\"video/mp4\"/>"
        f"Your browser does not support inline video. <a href=\"{video_rel}\">Download</a>"
        f"</video>"
    )
    if not video_abs.is_file():
        video_html = "<div class='placeholder'>(no video — side_by_side.mp4 missing)</div>"
    phmr_px = r.get('mean_reproj_mpjpe_phmr_px')
    b4d_px = r.get('mean_reproj_mpjpe_body4d_px')
    if phmr_px is not None and b4d_px is not None:
        if phmr_px < b4d_px * 0.95:
            winner_html = (
                f"<dd><b style='color:#2f855a'>{phmr_px:.2f} px PHMR</b> &middot; "
                f"<span style='color:#5b6770'>{b4d_px:.2f} px Body4D</span></dd>"
            )
        elif b4d_px < phmr_px * 0.95:
            winner_html = (
                f"<dd><b style='color:#2f855a'>{b4d_px:.2f} px Body4D</b> &middot; "
                f"<span style='color:#5b6770'>{phmr_px:.2f} px PHMR</span></dd>"
            )
        else:
            winner_html = (
                f"<dd>{phmr_px:.2f} px PHMR &middot; {b4d_px:.2f} px Body4D "
                f"<span style='color:#a0aec0'>(tied)</span></dd>"
            )
    else:
        winner_html = (
            f"<dd>{_fmt(phmr_px, '{:.2f}')} px PHMR &middot; {_fmt(b4d_px, '{:.2f}')} px Body4D</dd>"
        )
    meta = (
        f"<dl>"
        f"<dt>Frames compared</dt><dd>{r['n_frames']}</dd>"
        f"<dt>Dancers compared</dt><dd>{r['n_dancers']}</dd>"
        f"<dt>Raw MPJPE</dt><dd>{_fmt(r.get('mean_mpjpe_raw_m'))} m</dd>"
        f"<dt>PA-MPJPE</dt><dd>{_fmt(r.get('mean_mpjpe_pa_m'))} m</dd>"
        f"<dt>Jitter PHMR</dt><dd>{_fmt(r.get('mean_jitter_phmr_m'))} m/f</dd>"
        f"<dt>Jitter Body4D</dt><dd>{_fmt(r.get('mean_jitter_body4d_m'))} m/f</dd>"
        f"<dt>Foot-skating PHMR (world)</dt><dd>{_fmt(r.get('mean_foot_skating_phmr_world_m'), '{:.4f}')} m/f</dd>"
        f"<dt>Foot-skating Body4D (cam)</dt><dd>{_fmt(r.get('mean_foot_skating_body4d_cam_m'), '{:.4f}')} m/f</dd>"
        f"<dt>Reproj-vs-ViTPose</dt>{winner_html}"
        f"</dl>"
    )
    return (
        f"<section id='clip-{safe}' class='card'>"
        f"<h3>{safe}</h3>"
        f"<div class='clip'>"
        f"<div>{video_html}</div>"
        f"<div class='meta'>{meta}</div>"
        f"</div></section>"
    )


def build_html(rows: List[Dict], *, root: Path, title: str) -> str:
    sections = "\n".join(_build_clip_section(r, root) for r in rows)
    summary = _build_summary_table(rows)
    return (
        _HTML_HEAD.format(title=html_lib.escape(title))
        + f"<header><h1>{html_lib.escape(title)}</h1>"
          f"<div class='subtitle'>Dual 3D HMR pipeline — PromptHMR-Vid &times; SAM-Body4D &middot; {len(rows)} clip(s)</div></header>"
          f"<main>"
          f"<section class='card'><h2>Summary</h2>{summary}</section>"
          f"<section class='card'><h2>Glossary</h2>"
          f"<ul>"
          f"<li><b>MPJPE (raw)</b> — Mean Per-Joint Position Error (m); raw cam-frame distance between PHMR and Body4D.</li>"
          f"<li><b>PA-MPJPE</b> — MPJPE after per-dancer Procrustes alignment (rigid R+t, no scale); removes coordinate-frame mismatch.</li>"
          f"<li><b>Jitter</b> — mean inter-frame velocity (m/frame); lower = smoother.</li>"
          f"<li><b>Foot-skating PHMR (world)</b> — planted-foot velocity in PHMR's world frame, per-dancer floor calibration; lower = better contact.</li>"
          f"<li><b>Foot-skating Body4D (cam)</b> — same metric in cam frame; higher than PHMR-world because there's no floor reference.</li>"
          f"<li><b>Reproj-vs-ViTPose (PHMR &amp; Body4D, side-by-side)</b> — pixel error of each pipeline's reprojected COCO-17 vs the bundled ViTPose 2D in the native frame's coordinate system. <i>This is the metric that matches what you see on the side-by-side video.</i> Lower = mesh sits where the camera saw the dancer.</li>"
          f"</ul></section>"
          f"<section class='card'><h2>Per-clip details</h2>{sections}</section>"
          f"</main>"
        + _HTML_FOOT
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Path("runs/3d_compare"),
        help="Root directory containing per-clip subdirectories.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("runs/3d_compare/report.html"),
        help="Where to write the HTML file (relative video paths preserved).",
    )
    p.add_argument(
        "--title", default="Dual 3D HMR Pipeline — Operator Report",
        help="HTML <title> + page heading.",
    )
    args = p.parse_args(argv)

    root = args.root.expanduser().resolve()
    rows = aggregate_clip_metrics(root)
    print(f"[build_html_report] {len(rows)} clip(s) discovered under {root}")
    for r in rows:
        phmr = r.get('mean_reproj_mpjpe_phmr_px')
        b4d = r.get('mean_reproj_mpjpe_body4d_px')
        winner = ""
        if phmr is not None and b4d is not None:
            if phmr < b4d * 0.95:
                winner = " (PHMR wins)"
            elif b4d < phmr * 0.95:
                winner = " (Body4D wins)"
            else:
                winner = " (tied)"
        print(
            f"  - {r['name']}: T={r['n_frames']} N={r['n_dancers']} "
            f"PA={r.get('mean_mpjpe_pa_m')} "
            f"reproj_px PHMR={phmr} Body4D={b4d}{winner}"
        )
    html = build_html(rows, root=root, title=args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"[build_html_report] wrote {args.output} ({len(html):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
