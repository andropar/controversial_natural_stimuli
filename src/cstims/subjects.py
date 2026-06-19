"""Subject roster and cstim session helpers."""

from __future__ import annotations

from cstims import paths
from cstims.constants import CSTIM_SESSION_CANDIDATES, INPUT_SOURCE, SUBJECTS


def detect_available_sessions(subject: str) -> list[str]:
    """Detect which cstim sessions exist on disk for a subject."""
    glms_root = (
        paths.deepvision_fmri_root()
        / "derivatives/functional/1sTR_1pt5mm/glmsingle"
        / INPUT_SOURCE
        / subject
    )
    sessions = []
    for ses in CSTIM_SESSION_CANDIDATES:
        h5_path = glms_root / ses / "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
        if h5_path.exists():
            sessions.append(ses)
    return sessions


def parse_subject_arg(subject_arg: str) -> list[str]:
    """Parse a subject CLI argument into validated subject ids."""
    values = [item.strip() for item in str(subject_arg).split(",") if item.strip()]
    if not values or values == ["all"]:
        return list(SUBJECTS)
    if "all" in values:
        raise ValueError("'all' cannot be mixed with explicit subject ids")

    unknown = []
    for subject in values:
        if subject in SUBJECTS:
            continue
        if not detect_available_sessions(subject):
            unknown.append(subject)
    if unknown:
        raise ValueError(
            f"Unknown subject(s): {unknown}. Expected one of {SUBJECTS} or a subject "
            "with cstim sessions on disk."
        )
    return values


__all__ = [
    "CSTIM_SESSION_CANDIDATES",
    "INPUT_SOURCE",
    "SUBJECTS",
    "detect_available_sessions",
    "parse_subject_arg",
]
