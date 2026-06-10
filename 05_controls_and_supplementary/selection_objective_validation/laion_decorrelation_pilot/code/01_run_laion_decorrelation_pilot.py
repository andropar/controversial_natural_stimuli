#!/usr/bin/env python3
"""LAION pilot: noise-aware raw selection vs. pure raw decorrelation.

This version uses two model/layer feature spaces with existing subject encoding
models, so the same selected image sets can be evaluated in both fixed RSA
(raw feature RDMs) and mixed RSA (encoding-predicted RDMs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT = Path(__file__).resolve()
PILOT_DIR = SCRIPT.parents[1]
SHARE_ROOT = SCRIPT.parents[4]
SRC_DIR = SHARE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cstims.evaluation.constants import get_default_noise_level_multipliers
from cstims.evaluation.model_discrimination import model_discriminability
from cstims.feature_accessor import FeatureAccessor
from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor
from cstims.noise_estimation import rdm_noise_by_model
from cstims.rdm_cuda import get_rdm_vector
from cstims.selection import (
    TrackAggregationConfig,
    TrackDefinition,
    select_stimuli_multitrack,
)
from cstims.selection.primitives import compute_correlation_matrix

DEFAULT_DATA_DIR = Path(
    "/data/home_roth/_stachelschwein/rsa_based_selection/data/LAION_natural_sample"
)
DEFAULT_ENCODING_ROOT = (
    SHARE_ROOT
    / "01_brain_model_alignment/results/encoding_models/shared_subject_encoding_models/encoding_20251222_141301"
)
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
TWO_MODEL_CONFIGS = {
    "dinov2_vitl14": {
        "source": "deepjuice",
        "layer": "blocks.19.norm2",
        "aggregation": "cls",
        "encoding_layer_dir": "layerblocks_19_norm2",
        "source_feature_file": "features/dinov2_vitl14_layer-1.npy",
    },
    "slip_vit_l_slip": {
        "source": "deepjuice",
        "layer": "fc_norm",
        "aggregation": "cls",
        "encoding_layer_dir": "layerfc_norm",
        "source_feature_file": "features/slip_vit_l_slip_layer-1.npy",
    },
}
LAION11_MODEL_CONFIGS = {
    "dinov2_vitl14": {
        "source": "deepjuice",
        "layer": "blocks.19.norm2",
        "aggregation": "cls",
        "encoding_layer_dir": "layerblocks_19_norm2",
        "source_feature_file": "features/dinov2_vitl14_layer-1.npy",
    },
    "openclip_vit_so400m_14_siglip_webli": {
        "source": "deepjuice",
        "layer": "trunk.fc_norm",
        "aggregation": "cls",
        "encoding_layer_dir": "layertrunk_fc_norm",
        "source_feature_file": "features/openclip_vit_so400m_14_siglip_webli_layer-2.npy",
    },
    "robustness_imagenet_l2_eps3": {
        "source": "deepjuice",
        "layer": "model.avgpool",
        "aggregation": "flatten",
        "encoding_layer_dir": "layermodel_avgpool",
        "source_feature_file": "features/robustness_imagenet_l2_eps3_layer-2.npy",
    },
    "slip_vit_l_simclr": {
        "source": "deepjuice",
        "layer": "fc_norm",
        "aggregation": "cls",
        "encoding_layer_dir": "layerfc_norm",
        "source_feature_file": "features/slip_vit_l_simclr_layer-1.npy",
    },
    "slip_vit_l_slip": {
        "source": "deepjuice",
        "layer": "fc_norm",
        "aggregation": "cls",
        "encoding_layer_dir": "layerfc_norm",
        "source_feature_file": "features/slip_vit_l_slip_layer-1.npy",
    },
    "timm_vit_large_patch14_clip_quickgelu_224_openai": {
        "source": "deepjuice",
        "layer": "fc_norm",
        "aggregation": "cls",
        "encoding_layer_dir": "layerfc_norm",
        "source_feature_file": "features/timm_vit_large_patch14_clip_quickgelu_224_openai_layer-2.npy",
    },
    "torchvision_alexnet_imagenet1k_v1": {
        "source": "deepjuice",
        "layer": "classifier.5",
        "aggregation": "flatten",
        "encoding_layer_dir": "layerclassifier_5",
        "source_feature_file": "features/torchvision_alexnet_imagenet1k_v1_layer-2.npy",
    },
    "torchvision_resnet50_imagenet1k_v1": {
        "source": "deepjuice",
        "layer": "flatten",
        "aggregation": "flatten",
        "encoding_layer_dir": "layerflatten",
        "source_feature_file": "features/torchvision_resnet50_imagenet1k_v1_layer-2.npy",
    },
    "vicreg_resnet50": {
        "source": "deepjuice",
        "layer": "flatten",
        "aggregation": "flatten",
        "encoding_layer_dir": "layerflatten",
        "source_feature_file": "features/vicreg_resnet50_layer-1.npy",
    },
    "vissl_resnet50_mocov2": {
        "source": "deepjuice",
        "layer": "flatten",
        "aggregation": "flatten",
        "encoding_layer_dir": "layerflatten",
        "source_feature_file": "features/vissl_resnet50_mocov2_layer-1.npy",
    },
    "vissl_resnet50_supervised": {
        "source": "deepjuice",
        "layer": "flatten",
        "aggregation": "flatten",
        "encoding_layer_dir": "layerflatten",
        "source_feature_file": "features/vissl_resnet50_supervised_layer-1.npy",
    },
}
MODEL_SETS = {
    "two": TWO_MODEL_CONFIGS,
    "laion11": LAION11_MODEL_CONFIGS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--encoding-root", type=Path, default=DEFAULT_ENCODING_ROOT)
    parser.add_argument(
        "--model-set",
        choices=sorted(MODEL_SETS),
        default="two",
        help="Model set to use. 'laion11' is the full 10k LAION sample subset with compatible encoding weights.",
    )
    parser.add_argument("--target-size", type=int, default=50)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--corr-type", default="correlation")
    parser.add_argument("--noise-ceiling-target", type=float, default=0.46)
    parser.add_argument("--noise-calib-n", type=int, default=1000)
    parser.add_argument("--noise-calib-repeats", type=int, default=32)
    parser.add_argument("--eval-noise-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=PILOT_DIR)
    return parser.parse_args()


def load_image_table(data_dir: Path) -> pd.DataFrame:
    image_table = pd.read_csv(data_dir / "image_urls.csv")
    image_table["image_path"] = image_table["index"].map(
        lambda i: str(data_dir / "images" / f"{int(i)}.jpg")
    )
    return image_table


def extract_or_load_features(
    data_dir: Path,
    image_table: pd.DataFrame,
    output_dir: Path,
    model_configs: dict[str, dict[str, str]],
    device: torch.device,
    batch_size: int,
    force_extract: bool,
) -> dict[str, np.ndarray]:
    feature_dir = output_dir / "results" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    features: dict[str, np.ndarray] = {}

    for model_name, cfg in model_configs.items():
        out_path = feature_dir / f"{model_name}_features.npy"
        if out_path.exists() and not force_extract:
            arr = np.load(out_path, mmap_mode="r")
            print(f"Loaded cached {model_name} features: {arr.shape}")
            features[model_name] = arr
            continue

        source_feature_file = cfg.get("source_feature_file")
        if source_feature_file and not force_extract:
            source_path = data_dir / str(source_feature_file)
            if source_path.exists():
                arr = np.load(source_path, mmap_mode="r")
                if arr.shape[0] == len(image_table):
                    print(f"Loaded source cache for {model_name}: {arr.shape} ({source_path})")
                    features[model_name] = arr
                    continue

        print(f"Extracting {model_name} features...")
        extractor = UniversalFeatureExtractor(
            model_name=model_name,
            layer=cfg["layer"],
            aggregation=cfg["aggregation"],
            source=cfg["source"],
            device=device,
        )
        chunks: list[np.ndarray] = []
        paths = image_table["image_path"].tolist()
        for start in tqdm(range(0, len(paths), batch_size), desc=model_name):
            batch_paths = paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            tensors = [extractor.preprocess(img) for img in images]
            batch = torch.stack(tensors).to(device)
            with torch.inference_mode():
                feat = extractor.extract(batch)
            chunks.append(
                feat.detach()
                .cpu()
                .numpy()
                .reshape(len(batch_paths), -1)
                .astype(np.float32, copy=False)
            )
            del images, tensors, batch, feat
            if device.type == "cuda":
                torch.cuda.empty_cache()

        arr = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        np.save(out_path, arr)
        features[model_name] = np.load(out_path, mmap_mode="r")
        print(f"Saved {model_name} features: {arr.shape} -> {out_path}")

    n_rows = {name: arr.shape[0] for name, arr in features.items()}
    if len(set(n_rows.values())) != 1:
        raise ValueError(f"Feature row counts differ: {n_rows}")
    return features


def load_encoding_params(
    encoding_root: Path,
    subjects: list[str],
    model_names: list[str],
    model_configs: dict[str, dict[str, str]],
    device: torch.device,
    roi_subset: str = "hlvis",
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    params: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for subject in subjects:
        params[subject] = {}
        for model_name in model_names:
            layer_dir = model_configs[model_name]["encoding_layer_dir"]
            path = (
                encoding_root
                / f"{subject}_{model_name}.{layer_dir}"
                / "encoding_model.npz"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path) as z:
                W = z["weights"]
                b = z["intercept"]
                roi_key = f"roi_{roi_subset}"
                if roi_key in z:
                    mask = z[roi_key].astype(bool)
                    W = W[:, mask]
                    b = b[mask]
                params[subject][model_name] = {
                    "W": torch.from_numpy(W).to(device=device, dtype=torch.float32),
                    "bias": torch.from_numpy(b).to(device=device, dtype=torch.float32),
                }
    return params


def encode_for_subject(
    raw_features: dict[str, np.ndarray],
    model_names: list[str],
    indices: np.ndarray,
    subject_params: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = {}
    for model_name in model_names:
        feats = torch.from_numpy(np.asarray(raw_features[model_name][indices])).to(
            device=device, dtype=torch.float32
        )
        encoded[model_name] = (
            feats @ subject_params[model_name]["W"] + subject_params[model_name]["bias"]
        )
    return encoded


def calibrate_mixed_noise_by_subject(
    raw_features: dict[str, np.ndarray],
    model_names: list[str],
    encoding_params: dict[str, dict[str, dict[str, torch.Tensor]]],
    subjects: list[str],
    calib_n: int,
    metric: str,
    target_nc: float,
    corr_type: str,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    n_available = next(iter(raw_features.values())).shape[0]
    calib_indices = np.arange(min(calib_n, n_available), dtype=int)
    out: dict[str, dict[str, float]] = {}
    for subject in subjects:
        print(f"Calibrating mixed RDM-space noise for {subject}...")
        encoded = encode_for_subject(
            raw_features, model_names, calib_indices, encoding_params[subject], device
        )
        encoded_np = {name: encoded[name].detach().cpu().numpy() for name in model_names}
        out[subject] = rdm_noise_by_model(
            encoded_np,
            model_names,
            device,
            metric=metric,
            target_nc=target_nc,
            calib_n_examples=len(calib_indices),
            n_repeats=repeats,
            seed=seed,
            mode="numeric",
            corr_type=corr_type,
        )
        del encoded, encoded_np
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def select_variant(
    *,
    features: dict[str, np.ndarray],
    model_names: list[str],
    target_size: int,
    init_size: int,
    batch_size: int,
    metric: str,
    corr_type: str,
    seed: int,
    device: torch.device,
    var_noise_by_model: dict[str, float],
) -> np.ndarray:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    accessor = FeatureAccessor(
        features_by_model=features,
        model_names=model_names,
        pool_indices=np.arange(next(iter(features.values())).shape[0]),
        batch_size=batch_size,
        device=device,
        dtype=torch.float32,
    )
    result = select_stimuli_multitrack(
        raw_features=accessor,
        track_definitions=[
            TrackDefinition(
                name="raw",
                type="identity",
                var_noise_by_model=var_noise_by_model,
            )
        ],
        encoding_params_by_encoding={},
        track_aggregation=TrackAggregationConfig(
            norm_method="none", agg_method="identity"
        ),
        target_size=target_size,
        init_size=init_size,
        metric=metric,
        corr_type=corr_type,
        n_mc_samples=100,
        use_analytical=True,
        aggregation_within_models="mean",
        aggregation_across_models="min",
        device=device,
        image_filter=None,
        refine_max_passes=0,
        refine_min_replacements=0,
        checkpoint_dir=None,
        var_noise_raw=var_noise_by_model,
    )
    return result.current_indices.astype(int)


def rdms_from_numpy_features(
    features: dict[str, np.ndarray],
    model_names: list[str],
    indices: np.ndarray,
    metric: str,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [
            get_rdm_vector(
                torch.from_numpy(np.asarray(features[name][indices])).to(
                    device=device, dtype=torch.float32
                ),
                metric=metric,
            )
            for name in model_names
        ],
        dim=0,
    )


def rdms_from_tensor_features(
    features: dict[str, torch.Tensor],
    model_names: list[str],
    metric: str,
) -> torch.Tensor:
    return torch.stack([get_rdm_vector(features[name], metric=metric) for name in model_names], dim=0)


def evaluate_curve(
    *,
    method_name: str,
    rsa_space: str,
    subject: str,
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    noise_multipliers: np.ndarray,
    corr_type: str,
    n_noise_samples: int,
    seed: int,
) -> list[dict[str, float | str]]:
    device = rdms.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    rows = []

    clean_corr = compute_correlation_matrix(
        rdms.unsqueeze(0), rdms.unsqueeze(0), corr_type
    )[0]
    offdiag = clean_corr[
        ~torch.eye(clean_corr.shape[0], dtype=torch.bool, device=device)
    ]

    for multiplier in noise_multipliers:
        noised_rdms = (
            rdms
            + torch.randn(
                (n_noise_samples, *rdms.shape),
                device=device,
                generator=generator,
            )
            * noise_stds
            * float(multiplier)
        )
        scores = compute_correlation_matrix(
            rdms.repeat(n_noise_samples, 1, 1), noised_rdms, corr_type
        )
        disc = model_discriminability(scores)
        error_prob = float(disc["non_parametric_multiclass_error_prob"].item())
        rows.append(
            {
                "method": method_name,
                "rsa_space": rsa_space,
                "subject": subject,
                "noise_multiplier": float(multiplier),
                "snr": float(1.0 / multiplier),
                "accuracy": float(1.0 - error_prob),
                "error_prob": error_prob,
                "clean_offdiag_mean": float(offdiag.mean().item()),
                "clean_offdiag_max": float(offdiag.max().item()),
            }
        )
    return rows


def auc_over_log_noise(df: pd.DataFrame, y_col: str) -> float:
    ordered = df.sort_values("noise_multiplier")
    x = np.log10(ordered["noise_multiplier"].to_numpy(dtype=float))
    y = ordered[y_col].to_numpy(dtype=float)
    span = x[-1] - x[0]
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(y, x) / span)


def plot_curves(curves: pd.DataFrame, output_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    colors = {"noise_aware": "#0072B2", "pure_decorrelation": "#D55E00"}
    method_labels = {
        "noise_aware": "Noise-aware",
        "pure_decorrelation": "Pure decorrelation",
    }
    rsa_labels = {"fixed": "Fixed RSA", "mixed": "Mixed RSA"}
    linestyles = {"fixed": "-", "mixed": "--"}
    markers = {"fixed": "o", "mixed": "s"}

    plotted_values = []
    for rsa_space in ["fixed", "mixed"]:
        for method in ["noise_aware", "pure_decorrelation"]:
            sub = curves[
                (curves["rsa_space"] == rsa_space) & (curves["method"] == method)
            ].copy()
            if sub.empty:
                continue

            per_subject = (
                sub.groupby(["subject", "snr"], as_index=False)["accuracy"]
                .mean()
                .sort_values("snr")
            )
            mean_curve = (
                per_subject.groupby("snr", as_index=False)
                .agg(
                    accuracy=("accuracy", "mean"),
                    accuracy_sd=("accuracy", "std"),
                    n=("accuracy", "count"),
                )
                .sort_values("snr")
            )
            x = mean_curve["snr"].to_numpy(dtype=float)
            y = mean_curve["accuracy"].to_numpy(dtype=float)
            plotted_values.extend(y.tolist())

            if rsa_space == "mixed" and mean_curve["n"].max() > 1:
                sd = mean_curve["accuracy_sd"].fillna(0.0).to_numpy(dtype=float)
                ax.fill_between(
                    x,
                    np.clip(y - sd, 0.0, 1.0),
                    np.clip(y + sd, 0.0, 1.0),
                    color=colors[method],
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )

            ax.plot(
                x,
                y,
                color=colors[method],
                linestyle=linestyles[rsa_space],
                marker=markers[rsa_space],
                markersize=4.0,
                linewidth=2.0,
                label=f"{method_labels[method]} | {rsa_labels[rsa_space]}",
                zorder=3,
            )

    ax.axvline(1.0, color="0.25", lw=1.1, ls=(0, (4, 2)), zorder=2)
    ax.text(
        1.08,
        0.985,
        "empirical",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.009, 11.5)
    y_min = max(0.0, min(plotted_values) - 0.04) if plotted_values else 0.0
    ax.set_ylim(y_min, 1.015)
    ax.set_xlabel("Relative SNR (1 / noise multiplier)")
    ax.set_ylabel("Model recovery accuracy")
    ax.set_title(title)
    ax.grid(True, which="major", axis="both", lw=0.55, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"laion_decorrelation_pilot_curve.{ext}", dpi=200)
    plt.close(fig)


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rsa_space, method), sub in curves.groupby(["rsa_space", "method"]):
        mean_curve = (
            sub.groupby("noise_multiplier", as_index=False)
            .agg(
                error_prob=("error_prob", "mean"),
                accuracy=("accuracy", "mean"),
                clean_offdiag_mean=("clean_offdiag_mean", "mean"),
                clean_offdiag_max=("clean_offdiag_max", "mean"),
            )
        )
        empirical_pos = int(
            np.argmin(np.abs(mean_curve["noise_multiplier"].to_numpy(float) - 1.0))
        )
        empirical = mean_curve.iloc[empirical_pos]
        rows.append(
            {
                "rsa_space": rsa_space,
                "method": method,
                "error_auc": auc_over_log_noise(mean_curve, "error_prob"),
                "accuracy_auc": auc_over_log_noise(mean_curve, "accuracy"),
                "accuracy_at_noise_multiplier_1": float(empirical["accuracy"]),
                "error_at_noise_multiplier_1": float(empirical["error_prob"]),
                "clean_offdiag_mean": float(mean_curve["clean_offdiag_mean"].iloc[0]),
                "clean_offdiag_max": float(mean_curve["clean_offdiag_max"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    noise_vars: dict[str, float],
    model_names: list[str],
) -> None:
    empirical_lines = "\n".join(
        f"- {row.rsa_space} / {row.method}: {row.accuracy_at_noise_multiplier_1:.3f}"
        for row in summary.itertuples(index=False)
    )
    winners = (
        summary.sort_values(["rsa_space", "error_auc"])
        .groupby("rsa_space")
        .head(1)
    )
    winner_lines = "\n".join(
        f"- {row.rsa_space}: `{row.method}` (error AUC {row.error_auc:.4f})"
        for row in winners.itertuples(index=False)
    )
    model_lines = "\n".join(f"- `{name}`" for name in model_names)
    text = f"""# LAION Decorrelation Pilot

This pilot compares two raw-feature selection objectives on the 10k-image LAION
natural sample using {len(model_names)} encoder-compatible model feature spaces:

{model_lines}

Selection variants:

- `noise_aware`: project raw-track selector with calibrated RDM noise.
- `pure_decorrelation`: identical selector with zero RDM noise variance.

Evaluation spaces:

- `fixed`: raw-feature RDMs.
- `mixed`: subject encoding-predicted RDMs, averaged over {len(SUBJECTS)} subjects.

Configuration:

- target size: {args.target_size}
- init size: {args.init_size}
- seed: {args.seed}
- model set: `{args.model_set}`
- metric: `{args.metric}`
- correlation: `{args.corr_type}`
- target noise ceiling: {args.noise_ceiling_target}
- evaluation noise samples per point: {args.eval_noise_samples}

Raw calibrated RDM noise variances:

```json
{json.dumps(noise_vars, indent=2)}
```

Accuracy at empirical noise multiplier 1:

{empirical_lines}

Lower error AUC is better. Winners by evaluation space:

{winner_lines}
"""
    (output_dir / "REPORT.md").write_text(text)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    image_table = load_image_table(args.data_dir)
    model_configs = MODEL_SETS[args.model_set]
    features = extract_or_load_features(
        data_dir=args.data_dir,
        image_table=image_table,
        output_dir=output_dir,
        model_configs=model_configs,
        device=device,
        batch_size=args.extract_batch_size,
        force_extract=args.force_extract,
    )
    model_names = list(model_configs)

    print("Calibrating raw RDM-space noise...")
    noise_vars = rdm_noise_by_model(
        features,
        model_names,
        device,
        metric=args.metric,
        target_nc=args.noise_ceiling_target,
        calib_n_examples=args.noise_calib_n,
        n_repeats=args.noise_calib_repeats,
        seed=args.seed,
        mode="numeric",
        corr_type=args.corr_type,
    )
    zero_noise_vars = {name: 0.0 for name in model_names}

    selected_by_method = {}
    for method_name, variances in [
        ("noise_aware", noise_vars),
        ("pure_decorrelation", zero_noise_vars),
    ]:
        print(f"Selecting {method_name} stimuli...")
        selected_by_method[method_name] = select_variant(
            features=features,
            model_names=model_names,
            target_size=args.target_size,
            init_size=args.init_size,
            batch_size=args.batch_size,
            metric=args.metric,
            corr_type=args.corr_type,
            seed=args.seed,
            device=device,
            var_noise_by_model=variances,
        )

    selection_rows = []
    for method_name, indices in selected_by_method.items():
        for rank, idx in enumerate(indices):
            row = image_table.iloc[int(idx)].to_dict()
            selection_rows.append(
                {
                    "method": method_name,
                    "rank": rank,
                    "sample_index": int(idx),
                    "image_url_index": int(row.get("index", idx)),
                    "url": row.get("url", ""),
                    "image_path": row.get("image_path", ""),
                }
            )
    pd.DataFrame(selection_rows).to_csv(results_dir / "selection_indices.csv", index=False)

    noise_multipliers = get_default_noise_level_multipliers()
    raw_noise_stds = torch.tensor(
        [np.sqrt(noise_vars[name]) for name in model_names],
        device=device,
        dtype=torch.float32,
    ).view(len(model_names), 1)

    encoding_params = load_encoding_params(
        args.encoding_root, SUBJECTS, model_names, model_configs, device, roi_subset="hlvis"
    )
    mixed_noise_vars_by_subject = calibrate_mixed_noise_by_subject(
        raw_features=features,
        model_names=model_names,
        encoding_params=encoding_params,
        subjects=SUBJECTS,
        calib_n=args.noise_calib_n,
        metric=args.metric,
        target_nc=args.noise_ceiling_target,
        corr_type=args.corr_type,
        repeats=args.noise_calib_repeats,
        seed=args.seed,
        device=device,
    )

    curve_rows = []
    mixed_noise_rows = [
        {
            "rsa_space": "mixed",
            "model": name,
            "var_noise": mixed_noise_vars_by_subject[subject][name],
            "std_noise": float(np.sqrt(mixed_noise_vars_by_subject[subject][name])),
            "target_noise_ceiling": args.noise_ceiling_target,
            "subject": subject,
        }
        for subject in SUBJECTS
        for name in model_names
    ]
    for method_name, indices in selected_by_method.items():
        print(f"Evaluating {method_name} fixed curve...")
        raw_rdms = rdms_from_numpy_features(features, model_names, indices, args.metric, device)
        curve_rows.extend(
            evaluate_curve(
                method_name=method_name,
                rsa_space="fixed",
                subject="",
                rdms=raw_rdms,
                noise_stds=raw_noise_stds,
                noise_multipliers=noise_multipliers,
                corr_type=args.corr_type,
                n_noise_samples=args.eval_noise_samples,
                seed=args.seed + 10_000,
            )
        )

        for subject_idx, subject in enumerate(SUBJECTS):
            print(f"Evaluating {method_name} mixed curve for {subject}...")
            encoded = encode_for_subject(
                features, model_names, indices, encoding_params[subject], device
            )
            encoded_rdms = rdms_from_tensor_features(encoded, model_names, args.metric)
            encoded_noise_vars = mixed_noise_vars_by_subject[subject]
            encoded_noise_stds = torch.tensor(
                [np.sqrt(encoded_noise_vars[name]) for name in model_names],
                device=device,
                dtype=torch.float32,
            ).view(len(model_names), 1)
            curve_rows.extend(
                evaluate_curve(
                    method_name=method_name,
                    rsa_space="mixed",
                    subject=subject,
                    rdms=encoded_rdms,
                    noise_stds=encoded_noise_stds,
                    noise_multipliers=noise_multipliers,
                    corr_type=args.corr_type,
                    n_noise_samples=args.eval_noise_samples,
                    seed=args.seed + 20_000 + subject_idx,
                )
            )
            del encoded, encoded_rdms
            if device.type == "cuda":
                torch.cuda.empty_cache()

    curves = pd.DataFrame(curve_rows)
    curves.to_csv(results_dir / "noise_curves.csv", index=False)
    summary = summarize_curves(curves)
    summary.to_csv(results_dir / "summary_auc.csv", index=False)

    pd.DataFrame(
        [
            {
                "rsa_space": "fixed",
                "model": name,
                "var_noise": noise_vars[name],
                "std_noise": float(np.sqrt(noise_vars[name])),
                "target_noise_ceiling": args.noise_ceiling_target,
            }
            for name in model_names
        ]
        + mixed_noise_rows
    ).to_csv(results_dir / "noise_calibration.csv", index=False)

    metadata = {
        "data_dir": str(args.data_dir),
        "encoding_root": str(args.encoding_root),
        "model_configs": model_configs,
        "subjects": SUBJECTS,
        "config": {
            key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()
        },
        "device": str(device),
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    plot_curves(
        curves,
        figures_dir,
        f"LAION {len(model_names)}-model selection-objective pilot",
    )
    write_report(output_dir, args, summary, noise_vars, model_names)
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
