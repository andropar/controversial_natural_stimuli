# Leave-One-Subject-Out Selection Audit Feasibility

Existing leaderboard outputs cover raw-only, group-only, raw plus group-average encoding, raw plus each single-subject encoding, and raw plus all subject encodings.

No true leave-one-subject-out optimized selection output is present in `scripts/cursor/outputs/final_aggregate_plot/leaderboard_summary.csv` or in the final selection output directories inspected for this revision.

A true LOO audit would require rerunning the stimulus-selection optimizer five times per model set with one subject-specific encoding track omitted. It cannot be inferred from the available raw-plus-single-subject selections because those selections were independently optimized and do not represent the same objective with one track removed.
