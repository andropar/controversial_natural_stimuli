#!/usr/bin/env python3
"""Probe a model's layer names to choose early/mid/late candidates for the sweep.

Usage:
    python probe_layers.py <model_name> [--source deepjuice|custom] [--filter SUBSTRING]
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import argparse
import sys

import torch
from cstims.feature_extraction.universal_extractor import (
    UniversalFeatureExtractor, get_custom_model,
)

try:
    from deepjuice import get_deepjuice_model
except ImportError:
    get_deepjuice_model = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--source", default="deepjuice")
    p.add_argument("--filter", default=None,
                   help="Only show layer names containing this substring")
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--shapes", action="store_true",
                   help="Also probe activation shapes (slower)")
    args = p.parse_args()

    if args.source == "deepjuice":
        model, preprocess = get_deepjuice_model(args.model)
    else:
        model, preprocess = get_custom_model(args.model)
    if hasattr(model, "module"):
        model = model.module
    model.eval()

    print(f"\n=== {args.model} (source={args.source}) ===")
    print(f"top-level type: {type(model).__name__}")
    named = list(model.named_modules())
    print(f"total named_modules: {len(named)}")

    keep = named
    if args.filter:
        keep = [(n, m) for n, m in named if args.filter in n]
        print(f"after filter '{args.filter}': {len(keep)}")

    if args.shapes:
        # Probe a single dummy image
        dummy = torch.zeros(1, 3, 224, 224)
        try:
            dummy = preprocess(__import__("PIL").Image.new("RGB", (224, 224)))[None]
        except Exception:
            pass
        shapes = {}
        handles = []
        for name, mod in keep[: args.top]:
            def mkhook(name):
                def hk(_m, _i, o):
                    if isinstance(o, (list, tuple)):
                        o = o[-1]
                    if hasattr(o, "shape"):
                        shapes[name] = tuple(o.shape)
                return hk
            handles.append(mod.register_forward_hook(mkhook(name)))
        try:
            with torch.no_grad():
                model(dummy)
        finally:
            for h in handles:
                h.remove()
        for name, _ in keep[: args.top]:
            print(f"  {name:<60} {shapes.get(name, 'N/A')}")
    else:
        for name, mod in keep[: args.top]:
            print(f"  {name:<60} {type(mod).__name__}")


if __name__ == "__main__":
    main()
