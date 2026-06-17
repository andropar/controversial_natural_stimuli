#!/usr/bin/env python3
"""Compare between-model score spread at paper layer vs best/held-out layer.

The paper's claim: controversial stimuli widen between-model score spread
relative to the vicco baseline. We test whether "best layer per (subject,
model, set)" *also* collapses that spread, or whether spread is preserved.

Spread metric: median absolute pairwise difference of per-model means across
subjects (matches the existing rsa_large_benchmark figure).
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd

from cstims.paper.config import PAPER_ROOT
from layers_config import MAIN_LAYER

DATA_DIR = LAYER_SWEEP_ROOT / "results"
CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]


def median_pairwise_diff(x):
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan
    diffs = np.abs(x[:, None] - x[None, :])
    iu, ju = np.triu_indices_from(diffs, k=1)
    return float(np.median(diffs[iu, ju]))


def per_model_mean(df, mset, score_col, layer_col=None, layer_filter=None,
                  stim_type="controversial"):
    """Subject-averaged per-model score for a (set, stim_type)."""
    sub = df[(df["model_set"] == mset) & (df["stimulus_type"] == stim_type)]
    if layer_col and layer_filter:
        sub = sub[sub[layer_col] == layer_filter]
    return sub.groupby("model")[score_col].mean()


def main():
    fr = pd.read_csv(DATA_DIR / "fixed_rsa_layer_sweep.csv")
    mr = pd.read_csv(DATA_DIR / "wrsa_layer_sweep.csv")

    # Paper-layer subset
    fr_paper = fr[fr.apply(lambda r: r["layer"] == MAIN_LAYER[r["model"]], axis=1)]
    mr_paper = mr[mr.apply(lambda r: r["layer"] == MAIN_LAYER[r["model"]], axis=1)]

    # Best-layer (post-hoc per (subject, model, set))
    bl_fr = pd.read_csv(DATA_DIR / "best_layer_crsa_scores.csv").rename(columns={"crsa": "rsa"})
    bl_mr = pd.read_csv(DATA_DIR / "best_layer_wrsa_scores.csv").rename(columns={"wrsa_transfer": "rsa"})

    # Held-out best (mean of CV folds)
    ho = pd.read_csv(DATA_DIR / "held_out_rescue.csv")

    rows = []
    for mset in ["vicco"] + CSTIM_SETS:
        for label, src_paper, src_best, ho_col in [
            ("mRSA", mr_paper, bl_mr, "mrsa_best_mean"),
            ("fRSA", fr_paper, bl_fr, "frsa_best_mean"),
        ]:
            stim = "vicco" if mset == "vicco" else "controversial"
            # Use paper-layer model means
            paper_means = per_model_mean(src_paper, mset if mset != "vicco" else "vicco",
                                          "rsa", stim_type=stim)
            best_means = per_model_mean(src_best, mset if mset != "vicco" else "vicco",
                                         "rsa", stim_type=stim)
            ho_means = None
            if mset != "vicco":
                ho_sub = ho[ho["model_set"] == mset]
                if ho_col in ho.columns:
                    ho_means = ho_sub.groupby("model")[ho_col].mean()

            rows.append({
                "metric": label, "model_set": mset,
                "spread_paper": median_pairwise_diff(paper_means.values),
                "spread_best_biased": median_pairwise_diff(best_means.values),
                "spread_held_out": median_pairwise_diff(ho_means.values) if ho_means is not None and len(ho_means) > 1 else np.nan,
                "n_models": len(paper_means),
            })

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / "spread_summary.csv"
    out.to_csv(out_path, index=False)

    # Compute spread ratio = cstim_spread / vicco_spread per metric
    print("Between-model score spread (median absolute pairwise difference):\n")
    for metric in ["mRSA", "fRSA"]:
        m = out[out["metric"] == metric].set_index("model_set")
        if "vicco" not in m.index:
            continue
        v_paper = m.loc["vicco", "spread_paper"]
        v_best = m.loc["vicco", "spread_best_biased"]
        print(f"=== {metric} ===")
        print(f"vicco spread (paper-layer): {v_paper:.4f}")
        print(f"vicco spread (best-layer biased): {v_best:.4f}")
        print(f"{'set':<22}{'paper':>10}{'pl ratio':>10}"
              f"{'  best-biased':>14}{'bb ratio':>10}"
              f"{'  held-out':>11}{'ho ratio':>10}")
        for s in CSTIM_SETS:
            r = m.loc[s]
            r_pl = r["spread_paper"] / v_paper if v_paper else np.nan
            r_bb = r["spread_best_biased"] / v_best if v_best else np.nan
            r_ho = r["spread_held_out"] / v_paper if v_paper else np.nan  # held-out compared to vicco_paper
            print(f"{s:<22}{r['spread_paper']:>10.4f}{r_pl:>10.2f}"
                  f"{r['spread_best_biased']:>14.4f}{r_bb:>10.2f}"
                  f"{r['spread_held_out']:>11.4f}{r_ho:>10.2f}")
        print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
