# Agent Notes

## Python Environment

- Use the conda environment at `/data/home_roth/miniforge3` for this project.
- Prefer `/data/home_roth/miniforge3/bin/python` for Python commands.
- If using conda explicitly, use `conda run -p /data/home_roth/miniforge3 <command>`.
- For commands that import PyTorch/DeepJuice, set
  `LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}` so the
  conda `libstdc++` is used instead of the system copy.
- Do not rely on the system `/usr/bin/python3`; it is missing project dependencies such as pandas and scikit-learn.

## GitHub Pushes From Campus Hosts

- `~/.bashrc` exports the required campus proxy:
  `http_proxy=http://10.60.3.254:3128` and
  `https_proxy=http://10.60.3.254:3128`.
- HTTPS Git fetch/`ls-remote` works with those proxy variables, but HTTPS push
  can hang on credential prompts because `gh` is not authenticated here.
- SSH to `github.com:22` can hang. The verified push route is SSH over port 443
  to `ssh.github.com`, using `/home/roth/.ssh/id_ed25519_git` and an HTTP
  CONNECT proxy command through `10.60.3.254:3128`.
- The installed `/usr/bin/nc` does not support `-X connect`; use a small Python
  `ProxyCommand` or install a CONNECT-capable helper before running `git push`.
  Push URL form:
  `ssh://git@ssh.github.com:443/andropar/controversial_natural_stimuli.git`.


- When writing experiment code always clearly state ALL assumptions made about the experiment and associated methods. For example, when running a regression, make it explicit whether you assume that features are or are not standardized. 
- Before running experiments, perform a runtime test and estimate an ETA using synthetic data or small subsets of the real data. 
- Generally optimize for runtime performance - parallelize, batch, use GPU, compile with numba, etc. Look for opportunities to group data processing smartly. 
- When writing code for figures and creating them, ALWAYS look at the figure afterwards to check if labels are misaligned/overlapping the data, and also if the data looks "right" given the assumptions we're operating under. 
- Before creating files or changing files, ALWAYS communicate your plan on a high level to the user (e.g. "To do X I will create files Y and Z in directory Z and then run script W on them", or "To do X I will modify files Y and Z in directory Z and then run script W on them") and ASK FOR CONFIRMATION BEFORE PROCEEDING. 
