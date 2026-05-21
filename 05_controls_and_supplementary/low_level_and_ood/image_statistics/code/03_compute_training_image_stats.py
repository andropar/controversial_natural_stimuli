#!/usr/bin/env python3
"""
Compute low-level image statistics for the 1,492 "shared" DeepVision images
used to fit the encoding models that drove controversial-stimulus selection.

These images are stored inside stimuli_participant_p0X.hdf5 (JPEG bytes);
the metadata CSV flags them with `unique_or_shared == "shared"`. All five
subjects share an identical 1,492-image set, so we use p01.

Writes: appends rows with `stimulus_set="deepvision_train"` into
        data/image_stats.csv (replacing any existing rows for that set).
"""

import io
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# reuse the exact stat function from script 01
from importlib import util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("script01", _HERE / "01_compute_image_stats.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_stats = _mod.compute_stats

DEEPVISION_ROOT = Path("/SSD/jroth/deepvision_fmri")
STIMULI_HDF5 = DEEPVISION_ROOT / "stimuli_participant_p01.hdf5"
METADATA_CSV = DEEPVISION_ROOT / "metadata_p01.csv"
OUT_CSV = _HERE / "data" / "image_stats.csv"
SET_NAME = "deepvision_train"


def main():
    md = pd.read_csv(METADATA_CSV)
    shared_idx = md.index[md["unique_or_shared"] == "shared"].to_numpy()
    print(f"Shared images in metadata: {len(shared_idx)}")

    rows = []
    with h5py.File(STIMULI_HDF5, "r") as f:
        imgs = f["images"]
        for i in tqdm(shared_idx, desc="training"):
            im = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
            stats = compute_stats(im)
            stats.update(stimulus_set=SET_NAME, image=md.loc[i, "image_name"])
            rows.append(stats)

    new_df = pd.DataFrame(rows)

    # Merge with existing CSV: drop any prior rows for this set, then append.
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        existing = existing[existing["stimulus_set"] != SET_NAME]
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(new_df)} new rows ({SET_NAME}); file now {len(out)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
