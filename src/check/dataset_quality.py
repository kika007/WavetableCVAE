
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyrootutils
import torch
import torchaudio
from tqdm import tqdm


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

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
    args = parser.parse_args()

    wav_paths = sorted(args.dataset.glob(args.glob))
    if not wav_paths:
        raise SystemExit(f"No files matching '{args.glob}' found in {args.dataset}")

    dc_values: List[float] = []
    boundary_values: List[float] = []
    sample_rates = set()

    for wav_path in tqdm(wav_paths, desc="Scanning wavetables"):
        waveform, sr = torchaudio.load(wav_path)
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

if __name__ == "__main__":
    main()
