from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyrootutils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

DEFAULT_LABEL_DIR = root / "data/AKWF_44k1_600s" / "labels"
DEFAULT_ATTRS: Sequence[str] = (
    "dco_brightness",
    "dco_richness",
    "dco_oddenergy",
    "dco_zcr",
    "hardness",
)


def _load_rows(
    label_dir: Path, glob: str, attrs: Sequence[str]
) -> Tuple[np.ndarray, Dict[str, int], int]:
    """Load numeric rows for the requested attributes."""
    missing_counts: Dict[str, int] = {a: 0 for a in attrs}
    rows: List[List[float]] = []
    all_files = sorted(label_dir.glob(glob))

    for path in all_files:
        with open(path) as f:
            data = json.load(f)

        row: List[float] = []
        missing_attr = False
        for attr in attrs:
            val = data.get(attr, None)
            if isinstance(val, (int, float)):
                row.append(float(val))
            else:
                missing_counts[attr] += 1
                missing_attr = True
                break

        if not missing_attr:
            rows.append(row)

    return np.array(rows, dtype=float), missing_counts, len(all_files)


def _print_matrix(attrs: Sequence[str], corr: np.ndarray) -> None:
    width = max(12, max(len(a) for a in attrs) + 2)
    header = " " * (width + 2) + "".join(f"{a:>{width}}" for a in attrs)
    print(header)
    for name, row in zip(attrs, corr):
        row_str = "".join(f"{v:>{width}.3f}" for v in row)
        print(f"{name:>{width}}  {row_str}")


def _print_extremes(attrs: Sequence[str], corr: np.ndarray) -> None:
    """Report weakest and strongest links."""
    pairs = []
    for i in range(len(attrs)):
        for j in range(i + 1, len(attrs)):
            value = corr[i, j]
            pairs.append((attrs[i], attrs[j], value, abs(value)))

    weakest = min(pairs, key=lambda x: x[3])
    strongest = max(pairs, key=lambda x: x[3])
    most_negative = min(pairs, key=lambda x: x[2])

    print("\nSummary")
    print(f"- Lowest |corr|  : {weakest[0]} vs {weakest[1]} = {weakest[2]:+.3f}")
    print(f"- Highest |corr| : {strongest[0]} vs {strongest[1]} = {strongest[2]:+.3f}")
    print(f"- Most negative  : {most_negative[0]} vs {most_negative[1]} = {most_negative[2]:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute correlation matrix for psychoacoustic attributes stored in JSON labels."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_DIR, help="Folder containing label JSON files.")
    parser.add_argument("--glob", type=str, default="*.json", help="Glob pattern for label files.")
    parser.add_argument(
        "--attrs",
        nargs="+",
        default=list(DEFAULT_ATTRS),
        help="Attributes to correlate (must be numeric in JSON). Defaults to the dco_* set.",
    )
    parser.add_argument("--csv", type=Path, help="Optional path to save correlation matrix as CSV.")
    args = parser.parse_args()

    rows, missing_counts, total_files = _load_rows(args.labels, args.glob, args.attrs)
    if rows.size == 0:
        raise SystemExit("No complete rows found for the requested attributes.")

    used_files = rows.shape[0]
    corr = np.corrcoef(rows, rowvar=False)

    print("\nCorrelation matrix (Pearson)")
    print(f"- Label files scanned: {total_files}")
    print(f"- Files used (all attrs present): {used_files}")
    missing_summary = ", ".join(f"{k}:{v}" for k, v in missing_counts.items() if v > 0)
    print(f"- Missing counts: {missing_summary or 'none'}\n")

    _print_matrix(args.attrs, corr)
    _print_extremes(args.attrs, corr)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.csv,
            corr,
            delimiter=",",
            fmt="%.6f",
            header=",".join(args.attrs),
            comments="",
        )
        print(f"\nSaved CSV to {args.csv}")


if __name__ == "__main__":
    main()
