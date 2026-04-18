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
