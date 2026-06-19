from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CSTIMS_PATH_CONFIG", str(ROOT / "conf/paths/fusiform.env"))

from cstims.evaluation.teacher_student import (
    build_candidate_ops,
    run_track_rdm_recovery,
)


def _make_problem(eval_refit_mode: str):
    rng = np.random.default_rng(123)
    model_names = ["m0", "m1", "m2"]
    n_union = 28
    n_selected = 5
    n_features = 4
    random_target_union = {
        model: rng.normal(size=(n_union, n_features)).astype(np.float32)
        for model in model_names
    }
    selected_target = {
        model: rng.normal(size=(n_selected, n_features)).astype(np.float32)
        for model in model_names
    }
    random_subset_positions = [
        np.arange(12, 12 + n_selected, dtype=np.int64),
        np.arange(17, 17 + n_selected, dtype=np.int64),
    ]
    eval_raw = {"selected|0": selected_target}
    for subset_idx, positions in enumerate(random_subset_positions):
        eval_raw[f"random|{subset_idx}"] = {
            model: random_target_union[model][positions] for model in model_names
        }
    eval_meta = {
        "selected|0": ("selected", 0),
        "random|0": ("random", 0),
        "random|1": ("random", 1),
    }
    train_pos = np.arange(0, 8, dtype=np.int64)
    val_pos = np.arange(8, 12, dtype=np.int64)
    base_fit_pos = (
        np.concatenate([train_pos, val_pos])
        if eval_refit_mode == "eval_augmented_loo"
        else None
    )
    candidate_ops = build_candidate_ops(
        random_raw_union=random_target_union,
        eval_raw=eval_raw,
        refit_positions=np.arange(12, dtype=np.int64),
        train_pos=train_pos,
        val_pos=val_pos,
        base_fit_pos=base_fit_pos,
        model_names=model_names,
        alphas=[0.1, 1.0],
        eval_refit_mode=eval_refit_mode,
    )
    return {
        "model_set": "synthetic",
        "track": {"name": "raw", "type": "identity"},
        "refit_repeat_idx": 0,
        "selected_target": selected_target,
        "random_target_union": random_target_union,
        "random_subset_positions": random_subset_positions,
        "train_pos": train_pos,
        "val_pos": val_pos,
        "base_fit_pos": base_fit_pos,
        "eval_meta": eval_meta,
        "candidate_ops": candidate_ops,
        "model_names": model_names,
        "equivalence_labels": list(range(len(model_names))),
        "alphas": [0.1, 1.0],
        "noise_mults": np.asarray([0.5, 1.0], dtype=np.float64),
        "n_noise_samples": 2,
        "refit_train_n": len(train_pos),
        "refit_val_n": len(val_pos),
        "base_noise_ceiling": 0.46,
        "metric": "cosine",
        "corr_type": "spearman",
        "eval_noise_mode": "response",
        "fit_noise_calibration": "response",
        "eval_refit_mode": eval_refit_mode,
        "calibration_images": 4,
        "calibration_noise_samples": 1,
        "calibration_max_iter": 2,
        "target_dim": None,
        "teacher_cache_dir": None,
        "teacher_indices": None,
        "seed": 42,
        "batch_noise_samples": False,
    }


def _sorted_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    sort_columns = [
        "teacher_model",
        "noise_mult",
        "noise_sample_idx",
        "subset_type",
        "subset_idx",
    ]
    return df.sort_values(sort_columns).reset_index(drop=True)


class TeacherWorkerParityTest(unittest.TestCase):
    def test_teacher_workers_match_serial_rows(self) -> None:
        for eval_refit_mode in ["independent", "eval_augmented_loo"]:
            with self.subTest(eval_refit_mode=eval_refit_mode):
                kwargs = _make_problem(eval_refit_mode)
                serial = _sorted_rows(run_track_rdm_recovery(**kwargs, teacher_workers=1))
                parallel = _sorted_rows(run_track_rdm_recovery(**kwargs, teacher_workers=2))
                pd.testing.assert_frame_equal(
                    serial,
                    parallel,
                    check_exact=False,
                    atol=1e-12,
                    rtol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
