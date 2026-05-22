#!/usr/bin/env python3
"""Build ROI-level primary endpoint table from available masks/results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "13_roi_analysis" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    primary = pd.read_csv(PAPER / "03_statistics" / "results" / "primary_endpoint_summary.csv")
    rows = []
    for _, r in primary.iterrows():
        rows.append(
            {
                "subject": r["subject"],
                "roi": "hlvis",
                "model_set": r["model_set"],
                "metric": r["metric"],
                "baseline_type": r["baseline_type"],
                "score_cstim": r["score_cstim"],
                "score_baseline": r["score_baseline"],
                "delta": r["delta"],
                "score_cstim_NCnorm": r["score_cstim_NCnorm"],
                "score_baseline_NCnorm": r["score_baseline_NCnorm"],
                "delta_NCnorm": r["delta_NCnorm"],
                "spread_ratio": r["spread_ratio"],
                "status": "computed_from_primary_hlvis_pipeline",
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "roi_results.csv", index=False)

    note = OUT / "roi_feasibility_note.md"
    note.write_text(
        "# ROI Feasibility Note\n\n"
        "The current paper cache contains `visual_mask` and `hlvis_mask` in "
        "`01_brain_data/data/{subject}/voxel_metadata.npz`. It does not contain "
        "parcel labels or masks for early visual, ventral/object, or scene ROIs. "
        "Accordingly, `roi_results.csv` reports the primary hlvis endpoint only. "
        "Additional ROI splits require adding the atlas/parcel masks to the cache "
        "or rerunning `01_load_brain_data.py` with those masks exported.\n"
    )
    print(f"saved {OUT / 'roi_results.csv'}")
    print(f"saved {note}")


if __name__ == "__main__":
    main()
