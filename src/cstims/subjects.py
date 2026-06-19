"""Subject and session constants for cstims datasets."""

from __future__ import annotations


SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
CSTIM_SESSION_CANDIDATES = ["ses-32", "ses-33", "ses-34"]
INPUT_SOURCE = "tedana"


def parse_subject_arg(subject_arg: str) -> list[str]:
    """Parse a subject CLI argument into validated subject ids."""
    values = [item.strip() for item in str(subject_arg).split(",") if item.strip()]
    if not values or values == ["all"]:
        return list(SUBJECTS)

    unknown = [subject for subject in values if subject not in SUBJECTS]
    if unknown:
        raise ValueError(f"Unknown subject(s): {unknown}. Expected one of {SUBJECTS} or 'all'.")
    return values
