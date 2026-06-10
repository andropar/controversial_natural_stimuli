"""Feature extraction utilities that avoid deepjuice's activation hooks.

This module provides a :class:`UniversalFeatureExtractor` which relies on
``torchvision``'s feature extraction helpers when possible and falls back to
module-level forward hooks otherwise.  It supports models enumerated in
``00_stimulus_selection/resources/model_list.csv`` by delegating model loading and preprocessing
logic to :mod:`deepjuice` while handling the feature capture locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torchvision
from cornet import get_model
from torch import nn
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from transformers import AutoImageProcessor, AutoModel

from cstims.paths import resources_dir

try:  # pragma: no cover - exercised via monkeypatching in tests
    from deepjuice import get_deepjuice_model
except ImportError:  # pragma: no cover - the tests patch this symbol
    get_deepjuice_model = None  # type: ignore[assignment]

LayerSpec = Union[int, str]
TensorLike = Union[np.ndarray, torch.Tensor]

RESOURCES_DIR = resources_dir()

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def load_model_weights(model: nn.Module, model_path: str) -> nn.Module:
    # Robust loader for PyTorch >=2.6 checkpoints and assorted key formats
    import argparse

    import torch.serialization

    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass

    state = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    def strip_prefix(key: str) -> str:
        for pref in ("module.", "model.", "backbone."):
            if key.startswith(pref):
                return key[len(pref) :]
        return key

    sd = {strip_prefix(k): v for k, v in sd.items()}

    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    for key, param in model_state.items():
        if "num_batches_tracked" in key:
            continue
        if key in sd and getattr(sd[key], "shape", None) == param.shape:
            param.data.copy_(sd[key].data)
            loaded += 1
        else:
            skipped += 1
    print(f"loaded {loaded} layers; skipped {skipped} (missing/mismatch)")
    return model


def create_imagenet21k_model(
    model_name: str, num_classes: int = 10450, model_path: str = ""
) -> nn.Module:
    # Accept both 'imagenet21k_*' and '*_imagenet21k' conventions
    if model_name.startswith("imagenet21k_"):
        base_model_name = model_name.replace("imagenet21k_", "")
    elif model_name.endswith("_imagenet21k"):
        base_model_name = model_name.replace("_imagenet21k", "")
    else:
        base_model_name = model_name

    if base_model_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)
    elif base_model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif base_model_name == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    else:
        print(f"model: {model_name} not found !!")
        exit(-1)

    if not model_path:
        if base_model_name == "resnet50":
            model_path = RESOURCES_DIR / "resnet50_miil_21k.pth"
        elif base_model_name == "alexnet":
            model_path = RESOURCES_DIR / "alexnet_imagenet21k.pth"
        elif base_model_name == "vgg16":
            model_path = RESOURCES_DIR / "best_model_vgg16.pth"

    if model_path:
        model = load_model_weights(model, str(model_path))

    return model


def get_custom_model(
    model_name: str,
    model_path: Optional[str] = None,
) -> Tuple[nn.Module, Callable[[Any], torch.Tensor]]:
    # Accept both 'imagenet21k_*' and '*_imagenet21k' conventions
    if model_name == "vissl_resnet50_barlowtwins":
        model = torchvision.models.resnet50(weights=None)
        default_path = RESOURCES_DIR / "resnet50_barlowtwins.pth"
        load_model_weights(model, str(default_path))
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
    elif "imagenet21k" in model_name:
        model = create_imagenet21k_model(model_name, model_path="")
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )
    elif "cornet" in model_name:
        model = get_model("s", pretrained=True, map_location="cpu")
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(weights="IMAGENET1K_V1")
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
    elif model_name == "vgg16":
        model = torchvision.models.vgg16(weights="IMAGENET1K_V1")
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
    elif model_name == "dinov3-vitl16-pretrain-lvd1689m":
        model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
        processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )

        def preproc(pil_image) -> torch.Tensor:
            enc = processor(images=pil_image, return_tensors="pt")
            # [1, 3, H, W] -> [3, H, W]
            return enc["pixel_values"].squeeze(0)

        preprocess = preproc
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model, preprocess


@dataclass(frozen=True)
class LayerResolution:
    """Describes a resolved layer request."""

    name: str
    index: int
    strategy: str


class UniversalFeatureExtractor:
    """Extract intermediate representations from supported vision models.

    Parameters
    ----------
    model_name:
        Identifier used by :mod:`deepjuice` to load the network and its
        preprocessing pipeline.
    layer:
        Either an integer index or a string name identifying the desired
        activation.  Indices support negative indexing semantics.
    source:
        Model registry to consult.  ``"deepjuice"`` delegates to
        :mod:`deepjuice`, while ``"custom"`` reuses loaders from
        :mod:`cstims.feature_extraction.extractor`.
    model_path:
        Optional override path that will be forwarded to the loader when
        ``source`` requires it.
    device:
        Device on which the model should run.
    aggregation:
        How to aggregate multi-dimensional outputs. Options:
        - "cls": For 3D tensors (batch, tokens, features), select CLS token [:, 0]
        - "gap": For 4D tensors (batch, channels, H, W), use Global Average Pooling
        - "squeeze": For 4D tensors, squeeze spatial dimensions (only works if H=W=1)
        Default is "cls" for transformers, "squeeze" for conv nets (backward compatible).
    """

    _OUTPUT_KEY = "activation"

    def __init__(
        self,
        model_name: str,
        layer: LayerSpec,
        source: str = "deepjuice",
        model_path: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        aggregation: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.layer_spec = layer
        self.source = source
        self.device = torch.device(device)

        # Validate aggregation parameter
        if aggregation not in ("auto", "cls", "gap", "squeeze"):
            raise ValueError(
                f"aggregation must be one of 'auto', 'cls', 'gap', 'squeeze', got '{aggregation}'"
            )
        self.aggregation = aggregation

        if source == "deepjuice":
            if get_deepjuice_model is None:
                raise ImportError(
                    "deepjuice is required to load models but is not installed."
                )
            model, preprocess = get_deepjuice_model(model_name)
        elif source == "custom":
            model, preprocess = get_custom_model(model_name, model_path)
        else:
            raise ValueError(
                f"Unsupported source '{source}'. Expected one of {'deepjuice', 'custom'}."
            )

        if hasattr(model, "module"):
            model = model.module  # type: ignore[assignment]
        self.model: nn.Module = model.to(self.device).eval()
        self.preprocess: Callable = preprocess

        self._fx_nodes: Optional[List[str]] = None
        self._module_names: Optional[List[str]] = None
        self._target_module: Optional[nn.Module] = None
        self._feature_extractor: Optional[nn.Module] = None

        self.layer_resolution = self._initialise_layer(layer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, image_batch: TensorLike) -> torch.Tensor:
        """Run a forward pass and capture the configured activation."""

        inputs = self._prepare_tensor(image_batch)

        with torch.inference_mode():
            if self.layer_resolution.strategy == "fx":
                assert self._feature_extractor is not None
                outputs = self._feature_extractor(inputs)
                result = outputs[self._OUTPUT_KEY]
            else:
                assert self._target_module is not None
                activations: Dict[str, torch.Tensor] = {}

                def _hook(module: nn.Module, _: Tuple[torch.Tensor, ...], output):
                    if isinstance(output, (tuple, list)):
                        activations["value"] = output[-1].detach()
                    elif isinstance(output, dict):
                        # Prefer logits-like keys when available.
                        for key in ("logits", "out", "features"):
                            if key in output:
                                activations["value"] = output[key].detach()
                                break
                        else:
                            activations["value"] = list(output.values())[-1].detach()
                    else:
                        activations["value"] = output.detach()

                handle = self._target_module.register_forward_hook(_hook)
                try:
                    self.model(inputs)
                finally:
                    handle.remove()

                if "value" not in activations:
                    raise RuntimeError(
                        f"Hook on layer '{self.layer_resolution.name}' did not capture any activations."
                    )

                result = activations["value"]

            # Apply aggregation based on dimensionality and aggregation strategy
            aggregation_strategy = self.aggregation

            # Auto-detect strategy if "auto" is specified
            if aggregation_strategy == "auto":
                if result.ndim == 3:
                    aggregation_strategy = "cls"
                elif result.ndim == 4:
                    aggregation_strategy = "squeeze"
                else:
                    aggregation_strategy = None

            if aggregation_strategy == "cls" and result.ndim == 3:
                # For ViT-style outputs (batch, tokens, features), select CLS token
                result = result[:, 0]
            elif aggregation_strategy == "gap" and result.ndim == 4:
                # Global Average Pooling for conv feature maps
                # Handle both formats: (batch, channels, H, W) or (batch, H, W, channels)
                # Check which format by looking at typical channel counts vs spatial dims
                # Channels are usually > 64, spatial dims are usually <= 32 for 224x224 input
                if (
                    result.shape[1] > 64
                    and result.shape[2] <= 32
                    and result.shape[3] <= 32
                ):
                    # Format: (batch, channels, H, W) -> average over dims 2, 3
                    result = result.mean(dim=[2, 3])
                elif (
                    result.shape[3] > 64
                    and result.shape[1] <= 32
                    and result.shape[2] <= 32
                ):
                    # Format: (batch, H, W, channels) -> average over dims 1, 2
                    result = result.mean(dim=[1, 2])
                else:
                    # Ambiguous case: try to infer from shape
                    # If last dim is largest, assume it's channels
                    if result.shape[3] >= result.shape[1]:
                        result = result.mean(dim=[1, 2])  # (batch, H, W, channels)
                    else:
                        result = result.mean(dim=[2, 3])  # (batch, channels, H, W)
            elif aggregation_strategy == "squeeze" and result.ndim == 4:
                # Squeeze spatial dimensions (only works if H=W=1)
                result = result.squeeze(-1).squeeze(-1)
            elif result.ndim == 4:
                # If 4D but no valid aggregation, keep as-is (will be flattened downstream)
                pass

            return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialise_layer(self, layer: LayerSpec) -> LayerResolution:
        # Attempt FX-based extraction first.
        fx_nodes: Optional[List[str]] = None
        try:
            _, eval_nodes = get_graph_node_names(self.model)
            fx_nodes = list(eval_nodes)
        except Exception:
            fx_nodes = None

        if fx_nodes:
            try:
                resolved = self._resolve_layer(layer, fx_nodes)
            except (IndexError, ValueError):
                # Fall back to module traversal
                pass
            else:
                try:
                    feature_extractor = create_feature_extractor(
                        self.model,
                        return_nodes={resolved.name: self._OUTPUT_KEY},
                    ).to(self.device)
                except Exception:
                    # Some models (e.g. CORnet-S) cannot be traced via FX.
                    pass
                else:
                    self._feature_extractor = feature_extractor
                    self._fx_nodes = fx_nodes
                    return LayerResolution(resolved.name, resolved.index, "fx")

        # Fallback path using module hooks.
        module_names = self._collect_module_names(self.model)
        resolved = self._resolve_layer(layer, module_names)
        target_module = self._lookup_module(self.model, resolved.name)

        if target_module is None:
            raise ValueError(
                f"Unable to locate module '{resolved.name}' in model '{self.model_name}'."
            )

        self._module_names = module_names
        self._target_module = target_module
        return LayerResolution(resolved.name, resolved.index, "hook")

    def _prepare_tensor(self, image_batch: TensorLike) -> torch.Tensor:
        if isinstance(image_batch, np.ndarray):
            if image_batch.ndim == 3:
                image_batch = image_batch[np.newaxis, ...]
            if image_batch.ndim != 4:
                raise ValueError(f"Unexpected image_batch shape: {image_batch.shape}")

            processed: List[torch.Tensor] = []
            for i in range(image_batch.shape[0]):
                img = image_batch[i]
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                pil_image = self._to_pil_image(img)
                processed.append(self.preprocess(pil_image))
            tensor = torch.stack(processed)
        elif isinstance(image_batch, torch.Tensor):
            tensor = image_batch
        else:
            raise TypeError(
                "image_batch must be a numpy array or torch.Tensor, "
                f"got {type(image_batch)}"
            )

        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        return tensor

    @staticmethod
    def _resolve_layer(layer: LayerSpec, candidates: Iterable[str]) -> LayerResolution:
        names = list(candidates)
        if not names:
            raise ValueError(
                "No layers available to resolve the requested specification."
            )

        if isinstance(layer, int):
            index = layer if layer >= 0 else len(names) + layer
            if index < 0 or index >= len(names):
                raise IndexError(
                    f"Layer index {layer} out of range for {len(names)} available layers."
                )
            return LayerResolution(names[index], index, "")

        if isinstance(layer, str):
            if layer not in names:
                raise ValueError(
                    f"Layer name '{layer}' not found. Available layers: {names[:10]}..."
                )
            return LayerResolution(layer, names.index(layer), "")

        raise TypeError(f"layer must be specified as an int or str, got {type(layer)}")

    @staticmethod
    def _collect_module_names(model: nn.Module) -> List[str]:
        names = [name for name, _ in model.named_modules()]
        if not names:
            return [""]
        return names

    @staticmethod
    def _lookup_module(model: nn.Module, name: str) -> Optional[nn.Module]:
        if name == "":
            return model
        modules = dict(model.named_modules())
        return modules.get(name)

    @staticmethod
    def _summarize_output(output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return torch.Size(output.shape)
        if isinstance(output, (list, tuple)):
            return tuple(
                UniversalFeatureExtractor._summarize_output(item) for item in output
            )
        if isinstance(output, dict):
            return {
                key: UniversalFeatureExtractor._summarize_output(value)
                for key, value in output.items()
            }
        return None

    @staticmethod
    def _to_pil_image(img: np.ndarray):
        from PIL import Image

        if img.ndim == 2:
            mode = "L"
        elif img.shape[2] == 1:
            mode = "L"
            img = img[:, :, 0]
        elif img.shape[2] == 3:
            mode = "RGB"
        else:
            raise ValueError(f"Unsupported channel configuration: {img.shape}")

        return Image.fromarray(img, mode=mode)


__all__ = ["UniversalFeatureExtractor", "LayerResolution"]
