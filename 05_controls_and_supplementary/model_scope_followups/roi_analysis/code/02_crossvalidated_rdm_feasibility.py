#!/usr/bin/env python3
"""Document crossvalidated brain-RDM feasibility."""

from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "13_roi_analysis" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    note = OUT / "crossvalidated_brain_rdm_feasibility.md"
    note.write_text(
        "# Crossvalidated Brain-RDM Feasibility\n\n"
        "The CSTIMS cache includes averaged betas and per-stimulus repetition arrays "
        "(`cstim_betas_by_rep.npz`). It does not include run/session labels for each "
        "individual repetition in the paper-facing cache. Odd/even split-half RDMs "
        "are already used for noise ceilings, but a crossnobis or run-wise "
        "crossvalidated RDM requires independent run/session partitions and, "
        "ideally, run-wise noise covariance estimates. Because those labels are not "
        "available here, the crossvalidated brain-RDM robustness analysis is not run "
        "from this cache. Re-exporting repetition-level run/session metadata would "
        "make it feasible without collecting new data.\n"
    )
    print(f"saved {note}")


if __name__ == "__main__":
    main()
