"""Project-wide non-path constants.

Filesystem locations belong in :mod:`cstims.paths`; this module only contains
stable identifiers, model rosters, and numerical defaults.
"""

from __future__ import annotations

import numpy as np


SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
CSTIM_SESSION_CANDIDATES = ["ses-32", "ses-33", "ses-34"]
INPUT_SOURCE = "tedana"
ENCODING_MODE = "unique"

MODEL_SET_ORDER = [
    "all_models",
    "sota",
    "training_objective",
    "architecture",
    "dataset",
]

MODEL_SETS: dict[str, list[str]] = {
    "architecture": [
        "torchvision_vgg16_imagenet1k_v1",
        "torchvision_resnet50_imagenet1k_v1",
        "torchvision_convnext_base_imagenet1k_v1",
        "torchvision_vit_l_16_imagenet1k_v1",
        "cornet_s",
    ],
    "training_objective": [
        "vissl_resnet50_supervised",
        "vissl_resnet50_barlowtwins",
        "vissl_resnet50_mocov2",
        "vicreg_resnet50",
        "robustness_imagenet_l2_eps3",
    ],
    "sota": [
        "slip_vit_l_slip",
        "slip_vit_l_simclr",
        "timm_vit_large_patch14_clip_224_laion2b",
        "dinov2_vitl14",
        "openclip_vit_so400m_14_siglip_webli",
        "torchvision_convnext_base_imagenet1k_v1",
    ],
    "dataset": [
        "openclip_vit_l_14_quickgelu_metaclip_400m",
        "openclip_vit_l_14_quickgelu_metaclip_fullcc",
        "timm_vit_large_patch14_clip_224_dfn2b",
        "timm_vit_large_patch14_clip_quickgelu_224_openai",
        "openclip_vit_l_14_laion400m_e31",
    ],
}

MODEL_SETS["all_models"] = sorted(
    {model for models in MODEL_SETS.values() for model in models}
)

MODELS_EXCL_VICREG = [
    model for model in MODEL_SETS["all_models"] if model != "vicreg_resnet50"
]

MODEL_DISPLAY_NAMES = {
    "torchvision_vgg16_imagenet1k_v1": "VGG-16",
    "torchvision_resnet50_imagenet1k_v1": "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1": "ViT-L/16",
    "cornet_s": "CORnet-S",
    "vissl_resnet50_supervised": "Supervised",
    "vissl_resnet50_barlowtwins": "BarlowTwins",
    "vissl_resnet50_mocov2": "MoCoV2",
    "vicreg_resnet50": "VICReg",
    "robustness_imagenet_l2_eps3": "Robust-L2",
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m": "MetaCLIP-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": "MetaCLIP-Full",
    "timm_vit_large_patch14_clip_224_dfn2b": "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OpenAI",
    "openclip_vit_l_14_laion400m_e31": "CLIP-L400M",
    "torchvision_alexnet_imagenet1k_v1": "AlexNet",
}

RIDGE_ALPHAS = np.logspace(-2, 6, 50)

STIMULUS_SETS = [
    "vicco",
    "all_models",
    "architecture",
    "dataset",
    "sota",
    "training_objective",
]

EXPECTED_CSTIM_IMAGE_COUNTS = {
    "vicco": 292,
    "all_models": 100,
    "architecture": 100,
    "dataset": 100,
    "sota": 100,
    "training_objective": 100,
}
