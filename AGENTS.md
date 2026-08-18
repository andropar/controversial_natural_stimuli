# Agent Notes

## Communication

- Do not use the phrase "smoke test"; say "runtime validation run" or "minimal validation job" instead.
- Write in ASD-STE100 Simplified Technical English (STE).
- Don't use LaTeX equations, they don't render in the terminal.
- Do not repeat yourself in your answers.

## Python Environment

- Use the conda environment at `/data/home_roth/miniforge3` for this project.
- Prefer `/data/home_roth/miniforge3/bin/python` for Python commands.
- If using conda explicitly, use `conda run -p /data/home_roth/miniforge3 <command>`.
- For commands that import PyTorch/DeepJuice, set
  `LD_LIBRARY_PATH=/data/home_roth/miniforge3/lib:${LD_LIBRARY_PATH:-}` so the
  conda `libstdc++` is used instead of the system copy.
- Do not rely on the system `/usr/bin/python3`; it is missing project dependencies such as pandas and scikit-learn.

## Local Web Apps

- Never use Streamlit.
- Build small local labeling apps with Flask and separate HTML, CSS, and JavaScript files.

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
- Before running compute intensive experiments, perform a runtime test and estimate an ETA using synthetic data or small subsets of the real data. Ie. for simple plotting or short analysis (sub 5min) timing is not necessary.
- Generally optimize for runtime performance - parallelize, batch, use GPU, compile with numba, etc. Look for opportunities to group data processing smartly.
- When writing code for figures and creating them, ALWAYS look at the figure afterwards to check if labels are misaligned/overlapping the data, and also if the data looks "right" given the assumptions we're operating under.
- Before creating files or changing files, ALWAYS communicate your plan on a high level to the user (e.g. "To do X I will create files Y and Z in directory Z and then run script W on them", or "To do X I will modify files Y and Z in directory Z and then run script W on them") and ASK FOR CONFIRMATION BEFORE PROCEEDING.

- We had a server move - anything that was under /home/jroth or /SSD/jroth is now under /data/home_roth/_stachelschwein or /data/labshare/_stachelschwein/SSD/jroth respectively.

## Scanner-stimulus correction for the manuscript (2026-08-18)

- Manuscript analyses of the controversial-stimulus experiment must use the
  exact encoded image bytes in
  `/data/labshare/_stachelschwein/SSD/jroth/final_cstims_hdf5_files/stimuli.hdf5`.
  Its SHA-256 is
  `efb4a6e03a8ec3abe3e07121376405ff07e3a322f53312fca3d979c12299cbd4`.
- Use `metadata.csv` from the same directory and preserve its row order within
  each semantic stimulus group. Its SHA-256 is
  `cd8c7e31938dbd656aad40634ae1138d8f55a03dd868b3a1d3f981205c68953e`.
- Do not use the old folder-image copies for scanner-stimulus evaluation.
  They are retained as historical data, but they are not the authoritative
  images that were presented in the scanner.
- HDF5 group names are semantic. In the historical scanner brain cache only,
  the stored `architecture` and `dataset` labels are reversed. Swap these two
  labels exactly once when brain patterns are loaded. Do not swap image groups.
- For `robustness_imagenet_l2_eps3`, manuscript results use the epoch-105
  checkpoint at
  `/data/home_roth/_stachelschwein/.cache/torch/hub/robustness/imagenet_l2_3_0.pt`.
  Its SHA-256 is
  `2b9f420b0b1680ab1d4c77fb9006ab754d73683d4f2a6ac819de7fc563c58b7b`.
  The model is evaluated in evaluation mode with saved BatchNorm statistics.
- The frozen, validated manuscript plotting tables and their hash manifest are
  in `06_manuscript/figure_sources/data/scanner_hdf5_correction/`. Manuscript
  figure scripts must read these tables. Do not silently replace them with
  older analysis results.
- The old SRP-matched fixed-RSA result remains at
  `05_controls_and_supplementary/model_scope_followups/layer_sweep/results/frsa_best_shared_layer_transfer.csv`.
  The final manuscript Figure 3 uses native, unprojected fixed RSA at the layer
  selected by mixed RSA on independent shared images.
- The broad 118-model benchmark does not contain the affected Robust-L2 model.
  It was verified as complete for all five subjects and remains unchanged.
- Full provenance, assumptions, affected outputs, and validation commands are
  recorded in `06_manuscript/docs/scanner_stimulus_correction.md`.
