# ROI Feasibility Note

The current paper cache contains `visual_mask` and `hlvis_mask` in `01_brain_data/data/{subject}/voxel_metadata.npz`. It does not contain parcel labels or masks for early visual, ventral/object, or scene ROIs. Accordingly, `roi_results.csv` reports the primary hlvis endpoint only. Additional ROI splits require adding the atlas/parcel masks to the cache or rerunning `01_load_brain_data.py` with those masks exported.
