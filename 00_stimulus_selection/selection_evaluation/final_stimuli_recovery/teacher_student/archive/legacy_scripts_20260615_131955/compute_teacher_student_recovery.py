#!/usr/bin/env python3
"""Compatibility wrapper for the legacy teacher/student recovery CLI."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cstims.evaluation.teacher_student.recovery import main  # noqa: E402


if __name__ == "__main__":
    main()
