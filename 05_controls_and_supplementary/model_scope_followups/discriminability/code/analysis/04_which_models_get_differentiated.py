#!/usr/bin/env python3
"""Which models actually get differentiated by cstim?

Goes beyond the headline conversion rate to ask:
    1. Per-model conversion profile — which models are most often in
       converted pairs?
    2. Within-architecture-family vs across-family conversion — does cstim
       distinguish *same-architecture, different-training* models that
       vicco cannot?
    3. Effect-size shift — continuous distribution of |Δ_cstim| − |Δ_vicco|
       per pair, separated by outcome (converted, stayed tied, lost,
       remained-separated).
    4. Direction analysis — when a pair is separated on both, does cstim
       agree with vicco's rank order?
    5. Loss profile — 19 mRSA pairs lose discriminability on cstim. What
       structural pattern do they share?

Output:
    data/which_models_per_model_profile.csv
    data/which_models_family_conversion.csv
    data/which_models_pair_outcomes.csv (per-pair categorical outcome)
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[4]
PAPER = PROJECT / "experiments" / "cstim_paper"
sys.path.insert(0, str(PAPER))
from config import MODEL_SETS  # noqa

DATA_DIR = Path(__file__).resolve().parents[1] / "results"


# Architecture families: same backbone architecture, varying training.
# Within-family pairs test whether cstim can distinguish models that share
# architecture but differ in training objective / data.
ARCH_FAMILY = {
    # ResNet-50 family (6 models, all with the same conv backbone)
    "torchvision_resnet50_imagenet1k_v1":         "ResNet-50",
    "vissl_resnet50_supervised":                  "ResNet-50",
    "vissl_resnet50_barlowtwins":                 "ResNet-50",
    "vissl_resnet50_mocov2":                      "ResNet-50",
    "vicreg_resnet50":                            "ResNet-50",
    "robustness_imagenet_l2_eps3":                "ResNet-50",
    # ViT-L family (9 models, all ViT-Large variants)
    "torchvision_vit_l_16_imagenet1k_v1":               "ViT-L",
    "dinov2_vitl14":                                    "ViT-L",
    "slip_vit_l_slip":                                  "ViT-L",
    "slip_vit_l_simclr":                                "ViT-L",
    "timm_vit_large_patch14_clip_224_laion2b":          "ViT-L",
    "timm_vit_large_patch14_clip_224_dfn2b":            "ViT-L",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "ViT-L",
    "openclip_vit_l_14_quickgelu_metaclip_400m":        "ViT-L",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc":      "ViT-L",
    "openclip_vit_l_14_laion400m_e31":                  "ViT-L",
    # Singletons (no within-family pairs)
    "torchvision_vgg16_imagenet1k_v1":               "VGG",
    "torchvision_convnext_base_imagenet1k_v1":       "ConvNeXt",
    "cornet_s":                                      "CORnet",
    "openclip_vit_so400m_14_siglip_webli":           "ViT-So400m",
}

DISPLAY = {
    "torchvision_vgg16_imagenet1k_v1": "VGG-16",
    "torchvision_resnet50_imagenet1k_v1": "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1": "ViT-L/16",
    "cornet_s": "CORnet-S",
    "vissl_resnet50_supervised": "Sup",
    "vissl_resnet50_barlowtwins": "BarlowTw",
    "vissl_resnet50_mocov2": "MoCoV2",
    "vicreg_resnet50": "VICReg",
    "robustness_imagenet_l2_eps3": "Robust-L2",
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m": "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b": "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31": "CLIP-L400",
}


def label_outcome(row):
    tied_b = row["tied_baseline"]
    sep_c = row["separated_cstim"]
    near_c = row["near_ceiling"]
    if tied_b and sep_c:
        return "converted"           # tied baseline → separated cstim (gain)
    if tied_b and not sep_c:
        return "stayed_tied"         # tied baseline → tied cstim
    if not tied_b and sep_c:
        return "remained_separated"  # separated baseline → separated cstim
    return "lost"                     # separated baseline → tied cstim (loss)


def main():
    df = pd.read_csv(DATA_DIR / "pair_separation_mrsa.csv")
    df = df[df["model_set"] == "all_models"].copy()
    df["outcome"] = df.apply(label_outcome, axis=1)
    df["family_A"] = df["model_A"].map(ARCH_FAMILY)
    df["family_B"] = df["model_B"].map(ARCH_FAMILY)
    df["within_family"] = df["family_A"] == df["family_B"]
    df["family_label"] = df.apply(
        lambda r: r["family_A"] if r["within_family"] else "across", axis=1)
    df["effect_size_gain"] = df["mean_delta_cstim"].abs() - df["mean_delta_vicco"].abs()

    df.to_csv(DATA_DIR / "which_models_pair_outcomes.csv", index=False)

    # ---------------------------------------------------------------------
    # 1. Per-model conversion profile
    # ---------------------------------------------------------------------
    rows = []
    for m in MODEL_SETS["all_models"]:
        pairs_with_m = df[(df["model_A"] == m) | (df["model_B"] == m)]
        n_total_pairs = len(pairs_with_m)
        n_tied_b = int(pairs_with_m["tied_baseline"].sum())
        n_tied_b_nc = int(pairs_with_m["tied_and_near_ceiling"].sum())
        n_converted_nc = int((pairs_with_m["tied_and_near_ceiling"]
                              & pairs_with_m["separated_cstim"]).sum())
        n_lost = int(pairs_with_m["outcome"].eq("lost").sum())
        n_remained = int(pairs_with_m["outcome"].eq("remained_separated").sum())
        rows.append({
            "model": m, "display": DISPLAY.get(m, m),
            "family": ARCH_FAMILY.get(m, "?"),
            "n_pairs": n_total_pairs,
            "n_tied_baseline": n_tied_b,
            "n_tied_baseline_near_ceiling": n_tied_b_nc,
            "n_converted_near_ceiling": n_converted_nc,
            "n_lost": n_lost,
            "n_remained_separated": n_remained,
            "conversion_rate_nc": (100 * n_converted_nc / n_tied_b_nc) if n_tied_b_nc else np.nan,
            "loss_rate": 100 * n_lost / n_total_pairs,
        })
    per_model = pd.DataFrame(rows).sort_values("conversion_rate_nc", ascending=False)
    per_model.to_csv(DATA_DIR / "which_models_per_model_profile.csv", index=False)
    print("Per-model conversion profile (mRSA, all_models):")
    print(per_model[["display", "family", "n_tied_baseline_near_ceiling",
                      "n_converted_near_ceiling", "conversion_rate_nc",
                      "n_lost"]].to_string(index=False))

    # ---------------------------------------------------------------------
    # 2. Within-family vs across-family conversion
    # ---------------------------------------------------------------------
    rows = []
    for label in ["ResNet-50", "ViT-L", "across"]:
        if label == "across":
            sub = df[~df["within_family"]]
        else:
            sub = df[df["within_family"] & (df["family_A"] == label)]
        n_pairs = len(sub)
        n_tied_nc = int(sub["tied_and_near_ceiling"].sum())
        n_conv = int((sub["tied_and_near_ceiling"] & sub["separated_cstim"]).sum())
        n_lost = int(sub["outcome"].eq("lost").sum())
        n_remained = int(sub["outcome"].eq("remained_separated").sum())
        rows.append({
            "group": label, "n_pairs": n_pairs,
            "n_tied_near_ceiling": n_tied_nc,
            "n_converted": n_conv,
            "n_lost": n_lost,
            "n_remained_separated": n_remained,
            "conversion_rate_pct": (100 * n_conv / n_tied_nc) if n_tied_nc else np.nan,
        })
    family_df = pd.DataFrame(rows)
    family_df.to_csv(DATA_DIR / "which_models_family_conversion.csv", index=False)
    print("\nFamily conversion (mRSA, all_models):")
    print(family_df.to_string(index=False))

    # ---------------------------------------------------------------------
    # 3. Direction analysis (rank flips)
    # ---------------------------------------------------------------------
    df["sign_vicco"] = np.sign(df["mean_delta_vicco"])
    df["sign_cstim"] = np.sign(df["mean_delta_cstim"])
    df["direction_flipped"] = df["sign_vicco"] != df["sign_cstim"]
    flipped = df[df["direction_flipped"]]
    print(f"\nDirection flips (mRSA, all_models): "
          f"{len(flipped)}/{len(df)} pairs flip sign (vicco vs cstim mean)")
    sep_both = df[(~df["tied_baseline"]) & (df["separated_cstim"])]
    sep_both_flip = sep_both[sep_both["direction_flipped"]]
    print(f"Of {len(sep_both)} pairs separated on BOTH: "
          f"{len(sep_both_flip)} flipped sign (cstim contradicts baseline ranking)")

    # ---------------------------------------------------------------------
    # 4. Effect-size summary
    # ---------------------------------------------------------------------
    print("\nEffect-size shift |Δ_cstim| − |Δ_vicco| by outcome:")
    print(df.groupby("outcome")["effect_size_gain"]
          .agg(["count", "mean", "median", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
