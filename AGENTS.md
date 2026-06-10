# Agent Notes

## Python Environment

- Use the conda environment at `/data/home_roth/miniforge3` for this project.
- Prefer `/data/home_roth/miniforge3/bin/python` for Python commands.
- If using conda explicitly, use `conda run -p /data/home_roth/miniforge3 <command>`.
- For commands that import PyTorch/DeepJuice, set
  `LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}` so the
  conda `libstdc++` is used instead of the system copy.
- Do not rely on the system `/usr/bin/python3`; it is missing project dependencies such as pandas and scikit-learn.
