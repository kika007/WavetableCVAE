
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyrootutils
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from src.dataio.akwd_dataset import AKWDDataset

DEFAULT_DATASET = root / "data/AKWF_44k1_600s"
BOUNDARY_THRESHOLD = 0.05  # normalized |last - first| / peak  5%


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _summary(values: np.ndarray) -> Dict[str, float]:
    """Return small robust summary stats for a 1D array."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def analyze_waveform(waveform: torch.Tensor) -> Tuple[float, float]:
    """
    Compute key cleanliness metrics for a single waveform.

    Returns:
        dc_offset_abs_max: max absolute DC offset across channels.
        boundary_jump_norm_abs_max: max absolute boundary jump normalized by peak.
    """
    # waveform: (channels, samples)
    dc_offset = waveform.mean(dim=1)
    dc_offset_abs_max = dc_offset.abs().max().item()

    # boundary discontinuity: difference between end and start of loop
    boundary_jump = waveform[:, -1] - waveform[:, 0]

    # normalize by peak to make comparisons scale-free; clamp to avoid division by zero
    peak = waveform.abs().amax(dim=1).clamp_min(1e-9)
    boundary_jump_norm = boundary_jump / peak
    boundary_jump_norm_abs_max = boundary_jump_norm.abs().max().item()

    return dc_offset_abs_max, boundary_jump_norm_abs_max


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset cleanliness report (DC offset & boundary jump).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Folder containing .wav wavetables.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.wav",
        help="Glob pattern for files inside dataset folder.",
    )
    align_group = parser.add_mutually_exclusive_group()
    align_group.add_argument(
        "--align-start",
        dest="align_start",
        action="store_true",
        default=True,
        help="Roll each waveform so it starts at the first negative-to-positive zero-cross (matches dataset loader).",
    )
    align_group.add_argument(
        "--no-align-start",
        dest="align_start",
        action="store_false",
        help="Disable start alignment; analyze raw files.",
    )
    fix_group = parser.add_mutually_exclusive_group()
    fix_group.add_argument(
        "--fix-boundary",
        dest="fix_boundary",
        action="store_true",
        default=True,
        help="Linearly remove residual boundary jump so last == first (matches dataset loader).",
    )
    fix_group.add_argument(
        "--no-fix-boundary",
        dest="fix_boundary",
        action="store_false",
        help="Do not adjust residual boundary jump.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="If set, save waveform previews (after processing) for the first N files.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=8,
        help="Number of previews to save when --preview-dir is set.",
    )
    args = parser.parse_args()

    wav_paths = sorted(args.dataset.glob(args.glob))
    if not wav_paths:
        raise SystemExit(f"No files matching '{args.glob}' found in {args.dataset}")

    dc_values: List[float] = []
    boundary_values: List[float] = []
    sample_rates = set()

    saved_previews = 0

    def save_preview(waveform: torch.Tensor, sr: int, path: Path) -> None:
        """Save a simple time-domain plot of the processed waveform."""
        samples = waveform.shape[1]
        t = np.linspace(0, samples / sr, samples, endpoint=False)
        plt.figure(figsize=(6, 3))
        plt.plot(t, waveform[0].numpy(), label="ch0")
        if waveform.shape[0] > 1:
            plt.plot(t, waveform[1].numpy(), label="ch1", alpha=0.7)
        plt.title(path.stem)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        if waveform.shape[0] > 1:
            plt.legend()
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()
        nonlocal saved_previews
        saved_previews += 1

    for i, wav_path in enumerate(tqdm(wav_paths, desc="Scanning wavetables")):
        waveform, sr = torchaudio.load(wav_path)
        if args.align_start:
            waveform = AKWDDataset._align_start_to_zero(waveform)
        if args.fix_boundary:
            waveform = AKWDDataset._fix_boundary_discontinuity(waveform)
        waveform = AKWDDataset._remove_dc(waveform)

        if args.preview_dir is not None and i < args.preview_count:
            save_preview(waveform, sr, args.preview_dir / f"{wav_path.stem}.png")

        sample_rates.add(sr)
        dc_abs, boundary_abs = analyze_waveform(waveform)
        dc_values.append(dc_abs)
        boundary_values.append(boundary_abs)

    dc_array = np.array(dc_values)
    boundary_array = np.array(boundary_values)

    dc_stats = _summary(dc_array)
    boundary_stats = _summary(boundary_array)
    boundary_exceed = int(np.sum(boundary_array > BOUNDARY_THRESHOLD))

    print("\nDataset cleanliness")
    print(f"- Files analysed: {len(wav_paths)}")
    print(f"- Sample rates found: {sorted(sample_rates)}")
    print("\nDC offset (abs)")
    for k, v in dc_stats.items():
        print(f"  {k:>6}: {v:.6f}")

    print("\nBoundary jump |last - first| / peak")
    for k, v in boundary_stats.items():
        print(f"  {k:>6}: {v:.6f}")
    print(f"  above {BOUNDARY_THRESHOLD:.4f}: {boundary_exceed}")

    if args.preview_dir is not None:
        print(f"\nSaved {saved_previews} previews to {args.preview_dir} (limit {args.preview_count}).")
    else:
        print("\nTip: use --preview-dir PATH to save waveform PNG previews.")

if __name__ == "__main__":
    main()
