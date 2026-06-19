"""Extract multiple layers in a single forward pass.

Strategy:
    1. Try FX-based extraction with create_feature_extractor for ALL requested
       layers at once. This is the fastest path and works for ResNet-50,
       VGG-16, ConvNeXt-B, DINOv2.
    2. If FX tracing fails for any reason (e.g., CORnet's recurrent loops),
       fall back to registering one forward hook per requested layer and
       running a single forward pass.

Aggregation per layer:
    - "flatten": flatten all non-batch dimensions
    - "gap":     mean over spatial dims for 4D feature maps
    - "squeeze": squeeze trailing spatial dims (only valid if H=W=1)
    - "cls":     for 3D (B, T, D) take token 0

Outputs are torch tensors on CPU as float32 numpy arrays.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from cstims.feature_extraction.universal_extractor import get_custom_model

try:
    from deepjuice import get_deepjuice_model
except ImportError:
    get_deepjuice_model = None


def _aggregate(t: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply per-layer aggregation. Returns a 2D tensor (B, F).

    Permissive: if the requested mode does not match the input rank (e.g.
    'cls' on an already-pooled 2D output, or 'gap' on a 1D vector), pass
    through. This mirrors the existing UniversalFeatureExtractor's behavior
    — necessary because some paper layers like fc_norm / trunk.fc_norm /
    ln_post emit different ranks across model families.
    """
    if mode == "flatten":
        if t.ndim == 1:
            return t.unsqueeze(0)
        return t.reshape(t.shape[0], -1)
    if mode == "cls":
        if t.ndim == 3:
            return t[:, 0]
        if t.ndim == 2:
            return t
        # Higher-rank cls is undefined, flatten trailing dims.
        return t.reshape(t.shape[0], -1)
    if mode == "mean_patch":
        # Mean over the token/patch dimension. For models without CLS (e.g.
        # SigLIP). Expects (B, T, D) -> (B, D).
        if t.ndim == 3:
            return t.mean(dim=1)
        if t.ndim == 2:
            return t
        return t.reshape(t.shape[0], -1)
    if mode == "gap":
        if t.ndim == 4:
            # Heuristic: channels-first if dim1 > 64 and spatial dims small.
            if t.shape[1] > 64 and t.shape[2] <= 64 and t.shape[3] <= 64:
                return t.mean(dim=(2, 3))
            if t.shape[3] > 64 and t.shape[1] <= 64 and t.shape[2] <= 64:
                return t.mean(dim=(1, 2))
            return t.mean(dim=(2, 3))
        if t.ndim == 3:
            # Treat as (B, T, D) and pool tokens.
            return t.mean(dim=1)
        if t.ndim == 2:
            return t
        return t.reshape(t.shape[0], -1)
    if mode == "squeeze":
        if t.ndim == 2:
            return t
        if t.ndim == 4 and t.shape[2] == 1 and t.shape[3] == 1:
            return t.squeeze(-1).squeeze(-1)
        if t.ndim == 3 and t.shape[1] == 1:
            return t.squeeze(1)
        if t.ndim == 1:
            return t.unsqueeze(0)
        # Permissive flatten for unexpected shapes.
        return t.reshape(t.shape[0], -1)
    raise ValueError(f"unknown aggregation '{mode}'")


class MultiLayerExtractor:
    """Extract multiple layers in a single forward pass.

    Parameters
    ----------
    model_name : str
    source : str   "deepjuice" or "custom"
    layers : list of (layer_name, aggregation)
    device : str   default "cuda" if available else "cpu"
    """

    def __init__(
        self,
        model_name: str,
        source: str,
        layers: List[Tuple[str, str]],
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.source = source
        self.layers = list(layers)

        if source == "deepjuice":
            if get_deepjuice_model is None:
                raise ImportError("deepjuice required but not installed")
            model, preprocess = get_deepjuice_model(model_name)
        elif source == "custom":
            model, preprocess = get_custom_model(model_name)
        else:
            raise ValueError(f"unknown source '{source}'")

        if hasattr(model, "module"):
            model = model.module
        self.model: nn.Module = model.to(self.device).eval()
        self.preprocess: Callable = preprocess

        # Try FX path for all layers at once.
        self._fx_extractor = None
        self._hook_modules: Dict[str, nn.Module] = {}
        self._mode = None  # "fx" or "hooks"

        layer_names = [n for n, _ in self.layers]
        try:
            return_nodes = {name: name for name in layer_names}
            fx = create_feature_extractor(self.model, return_nodes=return_nodes)
            fx = fx.to(self.device).eval()
            # Verify with a tiny forward pass that we get all requested keys.
            with torch.inference_mode():
                probe = torch.zeros(1, 3, 224, 224, device=self.device)
                test_out = fx(probe)
                for name in layer_names:
                    if name not in test_out:
                        raise ValueError(f"FX did not return node '{name}'")
            self._fx_extractor = fx
            self._mode = "fx"
        except Exception:
            self._fx_extractor = None
            self._mode = None
        finally:
            # create_feature_extractor flips the underlying model to train()
            # whether or not it succeeds. Force back to eval. CORnet-S is
            # particularly sensitive to this (recurrent loops use buffer state).
            self.model.eval()

        if self._mode is None:
            # Hook fallback: every requested layer must be a real nn.Module.
            named = dict(self.model.named_modules())
            missing = [n for n in layer_names if n not in named]
            if missing:
                raise ValueError(
                    f"Cannot resolve layers {missing} for model {model_name}: "
                    f"not in FX graph and not in named_modules"
                )
            self._hook_modules = {n: named[n] for n in layer_names}
            self._mode = "hooks"

    def preprocess_images(self, pil_images: List) -> torch.Tensor:
        """Apply the model-specific CPU image transform and stack a batch."""
        tensors = [self.preprocess(img) for img in pil_images]
        return torch.stack(tensors)

    @torch.inference_mode()
    def extract_tensor_batch(self, batch: torch.Tensor) -> Dict[str, np.ndarray]:
        """Run a single forward pass from a preprocessed tensor batch.

        Returns
        -------
        dict[str, np.ndarray]   layer_name -> (B, F) float32 array
        """
        batch = batch.to(self.device, non_blocking=True)

        if self._mode == "fx":
            outputs = self._fx_extractor(batch)
        else:
            outputs: Dict[str, torch.Tensor] = {}
            handles = []
            try:
                for name, module in self._hook_modules.items():
                    def _make_hook(layer_name):
                        def _hook(_m, _inp, out):
                            if isinstance(out, (list, tuple)):
                                out = out[-1]
                            elif isinstance(out, dict):
                                for key in ("logits", "out", "features"):
                                    if key in out:
                                        out = out[key]
                                        break
                                else:
                                    out = list(out.values())[-1]
                            outputs[layer_name] = out.detach()
                        return _hook
                    handles.append(module.register_forward_hook(_make_hook(name)))
                self.model(batch)
            finally:
                for h in handles:
                    h.remove()

        result = {}
        for name, agg in self.layers:
            t = outputs[name]
            t = _aggregate(t, agg)
            result[name] = t.float().cpu().numpy()
        return result

    @torch.inference_mode()
    def extract(self, pil_images: List) -> Dict[str, np.ndarray]:
        """Run a single forward pass and return aggregated activations per layer."""
        return self.extract_tensor_batch(self.preprocess_images(pil_images))

    def free(self):
        """Release GPU memory."""
        del self.model
        del self._fx_extractor
        self._hook_modules.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
