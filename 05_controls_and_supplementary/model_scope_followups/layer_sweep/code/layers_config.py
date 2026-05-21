"""Per-model layer specifications for the layer sweep.

Each model has an ordered list of layers from early -> late.
Layer names are FX node names (or, for hook-fallback models, ``named_modules``
names). Aggregation per layer:
    - "flatten":     all non-batch dimensions are flattened; dense veRSA uses this
    - "gap":         spatial 4D feature maps -> global average pool
    - "squeeze":     already-1D or singleton-spatial outputs (passes through)
    - "cls":         transformer 3D outputs (B, T, D) -> token 0
    - "mean_patch":  3D (B, T, D) -> mean over tokens (no-CLS models, e.g. SigLIP)

The ``MAIN_LAYER`` mapping records the layer used in the existing
02_rsa_scores main pipeline (read from data/resources/model_list.csv) — that
layer must be present in MODEL_LAYERS for the sanity check.
"""

from typing import Dict, List, Tuple

LayerSpec = Tuple[str, str]  # (layer_name, aggregation)

MODEL_LAYERS: Dict[str, List[LayerSpec]] = {
    # ====================================================================
    # CONV NETS
    # ====================================================================
    "torchvision_resnet50_imagenet1k_v1": [
        ("relu",     "gap"),
        ("layer1",   "gap"),
        ("layer2",   "gap"),
        ("layer3",   "gap"),
        ("layer4",   "gap"),
        ("flatten",  "squeeze"),
    ],
    "torchvision_vgg16_imagenet1k_v1": [
        ("features.4",   "gap"),
        ("features.9",   "gap"),
        ("features.16",  "gap"),
        ("features.23",  "gap"),
        ("features.30",  "gap"),
        ("classifier.5", "squeeze"),
    ],
    "torchvision_convnext_base_imagenet1k_v1": [
        ("features.1.2",          "gap"),
        ("features.3.2",          "gap"),
        ("features.5.0.block.2",  "gap"),
        ("features.5.11.block.2", "gap"),
        ("features.5.23.block.2", "gap"),
        ("features.7.2",          "gap"),
    ],
    "cornet_s": [
        ("V1.output",       "gap"),
        ("V2.output",       "gap"),
        ("V4.output",       "gap"),
        ("IT.output",       "gap"),
        ("decoder.flatten", "squeeze"),
    ],
    # ResNet-50 self-supervised / robust variants — same architecture as
    # torchvision_resnet50, same layer scheme.
    "vissl_resnet50_supervised": [
        ("relu",    "gap"),
        ("layer1",  "gap"),
        ("layer2",  "gap"),
        ("layer3",  "gap"),
        ("layer4",  "gap"),
        ("flatten", "squeeze"),
    ],
    "vissl_resnet50_barlowtwins": [
        ("relu",    "gap"),
        ("layer1",  "gap"),
        ("layer2",  "gap"),
        ("layer3",  "gap"),
        ("layer4",  "gap"),
        ("flatten", "squeeze"),
    ],
    "vissl_resnet50_mocov2": [
        ("relu",    "gap"),
        ("layer1",  "gap"),
        ("layer2",  "gap"),
        ("layer3",  "gap"),
        ("layer4",  "gap"),
        ("flatten", "squeeze"),
    ],
    "vicreg_resnet50": [
        ("relu",    "gap"),
        ("layer1",  "gap"),
        ("layer2",  "gap"),
        ("layer3",  "gap"),
        ("layer4",  "gap"),
        ("flatten", "squeeze"),
    ],
    # robustness_imagenet_l2_eps3 wraps the ResNet-50 inside `.model.*`.
    # FX tracing fails (control flow). Hook-only. No flatten module.
    "robustness_imagenet_l2_eps3": [
        ("model.layer1",   "gap"),
        ("model.layer2",   "gap"),
        ("model.layer3",   "gap"),
        ("model.layer4",   "gap"),
        ("model.avgpool",  "squeeze"),
    ],

    # ====================================================================
    # ViT-L / TRANSFORMERS
    # ====================================================================
    "torchvision_vit_l_16_imagenet1k_v1": [
        ("encoder.layers.encoder_layer_3.add",  "cls"),
        ("encoder.layers.encoder_layer_7.add",  "cls"),
        ("encoder.layers.encoder_layer_11.add", "cls"),
        ("encoder.layers.encoder_layer_15.add", "cls"),
        ("encoder.layers.encoder_layer_19.add", "cls"),  # paper layer
        ("encoder.layers.encoder_layer_23.add", "cls"),
    ],
    "dinov2_vitl14": [
        ("blocks.3.norm2",  "cls"),
        ("blocks.7.norm2",  "cls"),
        ("blocks.11.norm2", "cls"),
        ("blocks.15.norm2", "cls"),
        ("blocks.19.norm2", "cls"),  # paper layer
        ("blocks.23.norm2", "cls"),
    ],
    # SLIP & timm CLIP-style ViT-L families: blocks.{0..23}, paper-specific
    # head named fc_norm or attn.qkv. Each entry is a CLS-pooled block
    # output, except the paper layer which keeps its specific aggregation.
    "slip_vit_l_slip": [
        ("blocks.3",  "cls"),
        ("blocks.7",  "cls"),
        ("blocks.11", "cls"),
        ("blocks.15", "cls"),
        ("blocks.19", "cls"),
        ("blocks.23", "cls"),
        ("fc_norm",   "cls"),  # paper layer
    ],
    "slip_vit_l_simclr": [
        ("blocks.3",  "cls"),
        ("blocks.7",  "cls"),
        ("blocks.11", "cls"),
        ("blocks.15", "cls"),
        ("blocks.19", "cls"),
        ("blocks.23", "cls"),
        ("fc_norm",   "cls"),  # paper layer
    ],
    "timm_vit_large_patch14_clip_224_laion2b": [
        ("blocks.3",            "cls"),
        ("blocks.7",            "cls"),
        ("blocks.11",           "cls"),
        ("blocks.15",           "cls"),
        ("blocks.18.attn.qkv",  "cls"),  # paper layer
        ("blocks.19",           "cls"),
        ("blocks.23",           "cls"),
    ],
    "timm_vit_large_patch14_clip_224_dfn2b": [
        ("blocks.3",            "cls"),
        ("blocks.7",            "cls"),
        ("blocks.11",           "cls"),
        ("blocks.15",           "cls"),
        ("blocks.18.attn.qkv",  "cls"),  # paper layer
        ("blocks.19",           "cls"),
        ("blocks.23",           "cls"),
    ],
    "timm_vit_large_patch14_clip_quickgelu_224_openai": [
        ("blocks.3",  "cls"),
        ("blocks.7",  "cls"),
        ("blocks.11", "cls"),
        ("blocks.15", "cls"),
        ("blocks.19", "cls"),
        ("blocks.23", "cls"),
        ("fc_norm",   "cls"),  # paper layer
    ],
    # OpenCLIP ViT-L: visual transformer at transformer.resblocks.{0..23},
    # final norm at ln_post.
    "openclip_vit_l_14_quickgelu_metaclip_400m": [
        ("transformer.resblocks.3",  "cls"),
        ("transformer.resblocks.7",  "cls"),
        ("transformer.resblocks.11", "cls"),
        ("transformer.resblocks.15", "cls"),
        ("transformer.resblocks.19", "cls"),
        ("transformer.resblocks.23", "cls"),
        ("ln_post",                   "cls"),  # paper layer
    ],
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": [
        ("transformer.resblocks.3",  "cls"),
        ("transformer.resblocks.7",  "cls"),
        ("transformer.resblocks.11", "cls"),
        ("transformer.resblocks.15", "cls"),
        ("transformer.resblocks.19", "cls"),
        ("transformer.resblocks.23", "cls"),
        ("ln_post",                   "cls"),  # paper layer
    ],
    "openclip_vit_l_14_laion400m_e31": [
        ("transformer.resblocks.3",  "cls"),
        ("transformer.resblocks.7",  "cls"),
        ("transformer.resblocks.11", "cls"),
        ("transformer.resblocks.15", "cls"),
        ("transformer.resblocks.19", "cls"),
        ("transformer.resblocks.23", "cls"),
        ("ln_post",                   "cls"),  # paper layer
    ],
    # SigLIP (so400m): trunk.blocks.{0..26} = 27 blocks, NO CLS token. Patch
    # tokens are mean-pooled by paper. paper layer trunk.fc_norm (Identity)
    # is post-pool (B, D).
    "openclip_vit_so400m_14_siglip_webli": [
        ("trunk.blocks.4",   "mean_patch"),
        ("trunk.blocks.9",   "mean_patch"),
        ("trunk.blocks.13",  "mean_patch"),
        ("trunk.blocks.17",  "mean_patch"),
        ("trunk.blocks.21",  "mean_patch"),
        ("trunk.blocks.26",  "mean_patch"),
        ("trunk.fc_norm",    "cls"),  # paper layer (already 2D)
    ],
}

MODEL_SOURCE = {
    # conv nets
    "torchvision_resnet50_imagenet1k_v1": "deepjuice",
    "torchvision_vgg16_imagenet1k_v1": "deepjuice",
    "torchvision_convnext_base_imagenet1k_v1": "deepjuice",
    "cornet_s": "custom",
    # resnet variants
    "vissl_resnet50_supervised": "deepjuice",
    "vissl_resnet50_barlowtwins": "custom",
    "vissl_resnet50_mocov2": "deepjuice",
    "vicreg_resnet50": "deepjuice",
    "robustness_imagenet_l2_eps3": "deepjuice",
    # transformers
    "torchvision_vit_l_16_imagenet1k_v1": "deepjuice",
    "dinov2_vitl14": "deepjuice",
    "slip_vit_l_slip": "deepjuice",
    "slip_vit_l_simclr": "deepjuice",
    "timm_vit_large_patch14_clip_224_laion2b": "deepjuice",
    "timm_vit_large_patch14_clip_224_dfn2b": "deepjuice",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "deepjuice",
    "openclip_vit_l_14_quickgelu_metaclip_400m": "deepjuice",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": "deepjuice",
    "openclip_vit_l_14_laion400m_e31": "deepjuice",
    "openclip_vit_so400m_14_siglip_webli": "deepjuice",
}

# The "main paper" layer per model: matches data/resources/model_list.csv
# and 02_rsa_scores. The sanity check verifies that fRSA at this layer
# reproduces the existing crsa_scores.csv values within CPU/GPU drift.
MAIN_LAYER = {
    "torchvision_resnet50_imagenet1k_v1":               "flatten",
    "torchvision_vgg16_imagenet1k_v1":                  "classifier.5",
    "torchvision_convnext_base_imagenet1k_v1":          "features.5.23.block.2",
    "cornet_s":                                         "decoder.flatten",
    "vissl_resnet50_supervised":                        "flatten",
    "vissl_resnet50_barlowtwins":                       "flatten",
    "vissl_resnet50_mocov2":                            "flatten",
    "vicreg_resnet50":                                  "flatten",
    "robustness_imagenet_l2_eps3":                      "model.avgpool",
    "torchvision_vit_l_16_imagenet1k_v1":               "encoder.layers.encoder_layer_19.add",
    "dinov2_vitl14":                                    "blocks.19.norm2",
    "slip_vit_l_slip":                                  "fc_norm",
    "slip_vit_l_simclr":                                "fc_norm",
    "timm_vit_large_patch14_clip_224_laion2b":          "blocks.18.attn.qkv",
    "timm_vit_large_patch14_clip_224_dfn2b":            "blocks.18.attn.qkv",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "fc_norm",
    "openclip_vit_l_14_quickgelu_metaclip_400m":        "ln_post",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc":      "ln_post",
    "openclip_vit_l_14_laion400m_e31":                  "ln_post",
    "openclip_vit_so400m_14_siglip_webli":              "trunk.fc_norm",
}


def _merge_layers(*layer_groups: List[LayerSpec]) -> List[LayerSpec]:
    merged: List[LayerSpec] = []
    seen = set()
    for group in layer_groups:
        for name, agg in group:
            if name in seen:
                continue
            merged.append((name, agg))
            seen.add(name)
    return merged


def _flatten_layers(layer_specs: List[LayerSpec]) -> List[LayerSpec]:
    return [(name, "flatten") for name, _ in layer_specs]


def _resnet50_dense(prefix: str = "") -> List[LayerSpec]:
    p = f"{prefix}." if prefix else ""
    layers: List[LayerSpec] = [
        (f"{p}conv1", "gap"),
        (f"{p}bn1", "gap"),
        (f"{p}relu", "gap"),
        (f"{p}maxpool", "gap"),
    ]
    for stage, count in (("layer1", 3), ("layer2", 4), ("layer3", 6), ("layer4", 3)):
        for i in range(count):
            block = f"{p}{stage}.{i}"
            layers.extend([
                (f"{block}.conv1", "gap"),
                (f"{block}.bn1", "gap"),
                (f"{block}.conv2", "gap"),
                (f"{block}.bn2", "gap"),
                (f"{block}.conv3", "gap"),
                (f"{block}.bn3", "gap"),
            ])
            if i == 0:
                layers.append((f"{block}.downsample", "gap"))
            layers.append((block, "gap"))
        layers.append((f"{p}{stage}", "gap"))
    if prefix:
        layers.append((f"{p}avgpool", "squeeze"))
    else:
        layers.append(("flatten", "squeeze"))
    return layers


def _vgg16_dense() -> List[LayerSpec]:
    layers = [(f"features.{i}", "gap") for i in range(31)]
    layers.extend((f"classifier.{i}", "squeeze") for i in (0, 3, 5))
    return layers


def _convnext_base_dense() -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    layers.extend((f"features.{i}", "gap") for i in (0, 2, 4, 6))
    for stage, count in (("features.1", 3), ("features.3", 3), ("features.5", 27), ("features.7", 3)):
        for i in range(count):
            block = f"{stage}.{i}"
            layers.extend([
                (f"{block}.block.0", "gap"),
                (f"{block}.block.2", "gap"),
                (f"{block}.block.3", "gap"),
                (f"{block}.block.4", "gap"),
                (f"{block}.block.5", "gap"),
            ])
            layers.append((f"{stage}.{i}", "gap"))
    return layers


def _torchvision_vit_l_dense() -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    for i in range(24):
        block = f"encoder.layers.encoder_layer_{i}"
        layers.extend([
            (f"{block}.ln", "cls"),
            (f"{block}.getitem", "cls"),
            (f"{block}.add", "cls"),
            (f"{block}.ln_1", "cls"),
            (f"{block}.mlp", "cls"),
            (f"{block}.add_1", "cls"),
        ])
    layers.append(("encoder.ln", "cls"))
    return layers


def _blocks_dense(prefix: str, n_blocks: int, agg: str = "cls", suffix: str = "") -> List[LayerSpec]:
    return [(f"{prefix}.{i}{suffix}", agg) for i in range(n_blocks)]


def _timm_vit_dense(prefix: str, n_blocks: int, agg: str = "cls") -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    for i in range(n_blocks):
        block = f"{prefix}.{i}"
        layers.extend([
            (f"{block}.norm1", agg),
            (f"{block}.attn.qkv", agg),
            (f"{block}.attn.proj", agg),
            (f"{block}.norm2", agg),
            (f"{block}.mlp.fc1", agg),
            (f"{block}.mlp.fc2", agg),
            (block, agg),
        ])
    return layers


def _dinov2_dense() -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    for i in range(24):
        block = f"blocks.{i}"
        layers.extend([
            (f"{block}.norm1", "cls"),
            (f"{block}.attn.qkv", "cls"),
            (f"{block}.attn.proj", "cls"),
            (f"{block}.norm2", "cls"),
            (f"{block}.mlp.fc1", "cls"),
            (f"{block}.mlp.fc2", "cls"),
            (block, "cls"),
        ])
    return layers


def _openclip_vit_dense() -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    for i in range(24):
        block = f"transformer.resblocks.{i}"
        layers.extend([
            (f"{block}.ln_1", "cls"),
            # MultiheadAttention owns an out_proj module, but OpenCLIP calls it
            # through torch.nn.functional, so forward hooks on attn.out_proj
            # never fire. ls_1 is the called post-attention branch output.
            (f"{block}.ls_1", "cls"),
            (f"{block}.ln_2", "cls"),
            (f"{block}.mlp.c_fc", "cls"),
            (f"{block}.mlp.c_proj", "cls"),
            (block, "cls"),
        ])
    layers.append(("ln_post", "cls"))
    return layers


def _cornet_s_dense() -> List[LayerSpec]:
    layers: List[LayerSpec] = [
        ("V1.conv1", "gap"),
        ("V1.norm1", "gap"),
        ("V1.nonlin1", "gap"),
        ("V1.pool", "gap"),
        ("V1.conv2", "gap"),
        ("V1.norm2", "gap"),
        ("V1.nonlin2", "gap"),
        ("V1.output", "gap"),
    ]
    for area, n_recurrences in (("V2", 2), ("V4", 4), ("IT", 2)):
        layers.extend([
            (f"{area}.conv_input", "gap"),
            (f"{area}.skip", "gap"),
            (f"{area}.norm_skip", "gap"),
            (f"{area}.conv1", "gap"),
            (f"{area}.conv2", "gap"),
            (f"{area}.conv3", "gap"),
        ])
        for t in range(n_recurrences):
            layers.extend([
                (f"{area}.norm1_{t}", "gap"),
                (f"{area}.norm2_{t}", "gap"),
                (f"{area}.norm3_{t}", "gap"),
            ])
        layers.append((f"{area}.output", "gap"))
    layers.extend([
        ("decoder.avgpool", "squeeze"),
        ("decoder.flatten", "squeeze"),
    ])
    return layers


# A denser but still tractable inventory for the expensive all-layer run.
# This means "all meaningful block/module outputs" rather than every FX op
# (matmuls, reshapes, dropout, residual adds, etc.). The hand-picked paper
# layers are merged in so the long run can reproduce the current comparisons.
DENSE_MODEL_LAYERS: Dict[str, List[LayerSpec]] = {
    "torchvision_resnet50_imagenet1k_v1": _merge_layers(
        _resnet50_dense(), MODEL_LAYERS["torchvision_resnet50_imagenet1k_v1"]
    ),
    "torchvision_vgg16_imagenet1k_v1": _merge_layers(
        _vgg16_dense(), MODEL_LAYERS["torchvision_vgg16_imagenet1k_v1"]
    ),
    "torchvision_convnext_base_imagenet1k_v1": _merge_layers(
        _convnext_base_dense(), MODEL_LAYERS["torchvision_convnext_base_imagenet1k_v1"]
    ),
    "cornet_s": _merge_layers(
        _cornet_s_dense(), MODEL_LAYERS["cornet_s"]
    ),
    "vissl_resnet50_supervised": _merge_layers(
        _resnet50_dense(), MODEL_LAYERS["vissl_resnet50_supervised"]
    ),
    "vissl_resnet50_barlowtwins": _merge_layers(
        _resnet50_dense(), MODEL_LAYERS["vissl_resnet50_barlowtwins"]
    ),
    "vissl_resnet50_mocov2": _merge_layers(
        _resnet50_dense(), MODEL_LAYERS["vissl_resnet50_mocov2"]
    ),
    "vicreg_resnet50": _merge_layers(
        _resnet50_dense(), MODEL_LAYERS["vicreg_resnet50"]
    ),
    "robustness_imagenet_l2_eps3": _merge_layers(
        _resnet50_dense("model"), MODEL_LAYERS["robustness_imagenet_l2_eps3"]
    ),
    "torchvision_vit_l_16_imagenet1k_v1": _merge_layers(
        _torchvision_vit_l_dense(), MODEL_LAYERS["torchvision_vit_l_16_imagenet1k_v1"]
    ),
    "dinov2_vitl14": _merge_layers(
        _dinov2_dense(), MODEL_LAYERS["dinov2_vitl14"]
    ),
    "slip_vit_l_slip": _merge_layers(
        _timm_vit_dense("blocks", 24), MODEL_LAYERS["slip_vit_l_slip"]
    ),
    "slip_vit_l_simclr": _merge_layers(
        _timm_vit_dense("blocks", 24), MODEL_LAYERS["slip_vit_l_simclr"]
    ),
    "timm_vit_large_patch14_clip_224_laion2b": _merge_layers(
        _timm_vit_dense("blocks", 24), MODEL_LAYERS["timm_vit_large_patch14_clip_224_laion2b"]
    ),
    "timm_vit_large_patch14_clip_224_dfn2b": _merge_layers(
        _timm_vit_dense("blocks", 24), MODEL_LAYERS["timm_vit_large_patch14_clip_224_dfn2b"]
    ),
    "timm_vit_large_patch14_clip_quickgelu_224_openai": _merge_layers(
        _timm_vit_dense("blocks", 24), MODEL_LAYERS["timm_vit_large_patch14_clip_quickgelu_224_openai"]
    ),
    "openclip_vit_l_14_quickgelu_metaclip_400m": _merge_layers(
        _openclip_vit_dense(), MODEL_LAYERS["openclip_vit_l_14_quickgelu_metaclip_400m"]
    ),
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": _merge_layers(
        _openclip_vit_dense(), MODEL_LAYERS["openclip_vit_l_14_quickgelu_metaclip_fullcc"]
    ),
    "openclip_vit_l_14_laion400m_e31": _merge_layers(
        _openclip_vit_dense(), MODEL_LAYERS["openclip_vit_l_14_laion400m_e31"]
    ),
    "openclip_vit_so400m_14_siglip_webli": _merge_layers(
        _timm_vit_dense("trunk.blocks", 27, agg="mean_patch"),
        MODEL_LAYERS["openclip_vit_so400m_14_siglip_webli"],
    ),
}

LAYER_SETS: Dict[str, Dict[str, List[LayerSpec]]] = {
    "configured": MODEL_LAYERS,
    "dense": {model: _flatten_layers(specs) for model, specs in DENSE_MODEL_LAYERS.items()},
}


def get_layer_set(name: str = "configured") -> Dict[str, List[LayerSpec]]:
    if name not in LAYER_SETS:
        raise ValueError(f"Unknown layer set '{name}'. Options: {sorted(LAYER_SETS)}")
    return LAYER_SETS[name]

LATE_LAYER = MAIN_LAYER  # backwards-compat alias


def layer_depth_rank(model: str, layer_name: str) -> int:
    names = [n for n, _ in MODEL_LAYERS[model]]
    return names.index(layer_name)


def layer_depth_frac(model: str, layer_name: str) -> float:
    names = [n for n, _ in MODEL_LAYERS[model]]
    if len(names) == 1:
        return 1.0
    return names.index(layer_name) / (len(names) - 1)


STIMULUS_SETS = ["vicco", "all_models", "architecture", "dataset", "sota", "training_objective"]
