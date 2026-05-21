from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from ..data_loader import model_list


EncodingName = str
ModelName = str


EncodingParams = Dict[str, torch.Tensor]
EncodingParamsByModel = Dict[ModelName, EncodingParams]
EncodingParamsByEncoding = Dict[EncodingName, EncodingParamsByModel]


def _sanitize_layer_name(layer: str | int) -> str:
    layer_str = str(layer).strip()
    return (
        layer_str.replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_encoding_params_by_encoding(
    encoding_root: Path,
    model_list_csv: Path,
    encoding_names: Iterable[EncodingName],
    device: torch.device,
    roi_subset: str | None = "hlvis",
) -> EncodingParamsByEncoding:
    """
    Load linear encoding parameters (W, bias) for multiple encoding models.

    Supports the format from fit_encoding_hydra.py:

        encoding_root / f\"{encoding_name}_{model}.layer{layer_safe}\" / \"encoding_model.npz\"

    The npz file should contain 'weights' and 'intercept' keys.
    Weights are expected to be in raw feature space (already rescaled).

    Args:
        encoding_root: Directory containing encoding model subdirectories.
        model_list_csv: CSV file with model names and layers.
        encoding_names: List of encoding names to load.
        device: Torch device to load tensors to.
        roi_subset: If specified, subset output voxels to this ROI (e.g., "hlvis").
                    The npz file must contain a "roi_{roi_subset}" boolean mask.
                    Set to None to use all voxels.
    """
    encoding_root = encoding_root.resolve()
    model_list_csv = model_list_csv.resolve()

    all_model_names, all_layer_names = model_list(model_list_csv)
    if len(all_model_names) != len(all_layer_names):
        raise ValueError("model_list returned mismatched names and layers.")

    params_by_encoding: EncodingParamsByEncoding = {}

    for enc_name in encoding_names:
        per_model: EncodingParamsByModel = {}
        roi_mask: np.ndarray | None = None

        for model_name, layer in zip(all_model_names, all_layer_names):
            layer_safe = _sanitize_layer_name(layer)
            model_dir = encoding_root / f"{enc_name}_{model_name}.layer{layer_safe}"
            enc_npz = model_dir / "encoding_model.npz"
            if not enc_npz.exists():
                logging.warning(
                    f"Encoding file missing for encoding='{enc_name}', model='{model_name}': {enc_npz}"
                )
                continue

            with np.load(enc_npz) as z:
                W_raw = z["weights"]
                b_raw = z["intercept"]

                # Load ROI mask from first model (should be same for all models in encoding)
                if roi_subset is not None and roi_mask is None:
                    roi_key = f"roi_{roi_subset}"
                    if roi_key in z:
                        roi_mask = z[roi_key].astype(bool)
                        logging.info(
                            f"Encoding '{enc_name}': subsetting to ROI '{roi_subset}' "
                            f"({roi_mask.sum()}/{len(roi_mask)} voxels)"
                        )
                    else:
                        logging.warning(
                            f"Encoding '{enc_name}': ROI '{roi_subset}' not found in {enc_npz}, "
                            f"using all voxels. Available keys: {list(z.keys())}"
                        )

                # Apply ROI mask if available
                if roi_mask is not None:
                    W_raw = W_raw[:, roi_mask]
                    b_raw = b_raw[roi_mask]

                W_tensor = torch.from_numpy(W_raw).to(device=device, dtype=torch.float32)
                b_tensor = torch.from_numpy(b_raw).to(device=device, dtype=torch.float32)

            per_model[model_name] = {"W": W_tensor, "bias": b_tensor}

        if len(per_model) != len(all_model_names):
            logging.warning(
                f"Encoding '{enc_name}': only {len(per_model)}/{len(all_model_names)} models found, skipping"
            )
            continue

        params_by_encoding[enc_name] = per_model

    return params_by_encoding


def encode_batch_for_all_encodings(
    raw_batch: Dict[ModelName, torch.Tensor],
    params_by_encoding: EncodingParamsByEncoding,
) -> Dict[EncodingName, Dict[ModelName, torch.Tensor]]:
    """
    Apply all encoding models to a batch of raw features.

    Args:
        raw_batch: Dict[model_name, Tensor[B, D_in]] on device.
        params_by_encoding: Nested dict encoding_name -> model_name -> {\"W\", \"bias\"}.

    Returns:
        encoded: Dict[encoding_name, Dict[model_name, Tensor[B, D_out]]]
    """
    if not raw_batch:
        raise ValueError("raw_batch is empty.")

    encoded: Dict[EncodingName, Dict[ModelName, torch.Tensor]] = {}

    for enc_name, per_model_params in params_by_encoding.items():
        encoded_per_model: Dict[ModelName, torch.Tensor] = {}
        for model_name, feats in raw_batch.items():
            if model_name not in per_model_params:
                raise KeyError(
                    f"Missing encoding params for model '{model_name}' in '{enc_name}'."
                )
            W = per_model_params[model_name]["W"]
            b = per_model_params[model_name]["bias"]

            # Ensure params are on the same device as features
            if W.device != feats.device:
                W = W.to(device=feats.device, dtype=torch.float32, non_blocking=True)
            if b.device != feats.device:
                b = b.to(device=feats.device, dtype=torch.float32, non_blocking=True)

            encoded_per_model[model_name] = feats @ W + b

        encoded[enc_name] = encoded_per_model

    return encoded


