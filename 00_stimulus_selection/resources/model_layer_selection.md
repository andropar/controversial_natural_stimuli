# Model Layer Selection Guide

---

## ResNet-50 Family (supervised, MoCo v2, VICReg)

**Why this choice:**  
For ResNets, the classifier head is the final `fc` layer. The representation right before that is the global-average-pooled 2048-d feature. This is the closest non-task layer, widely used as the “penultimate” feature, and avoids mixing in head/class information.

### vissl_resnet50_supervised

- **Pick:** `flatten` → shape `[1, 2048]`  
  (Same tensor as `avgpool` after flattening; sits immediately before `fc`.)

- **Good alternates:**
  - `avgpool` → `[1, 2048, 1, 1]` (flatten yourself)
  - If you want a spatial last-conv map: `layer4.2.relu_2` → `[1, 2048, 7, 7]`

---

### vissl_resnet50_mocov2

- **Pick:** `flatten` → `[1, 2048]`  
  (MoCo’s projector/predictor aren’t in this trace; the pooled trunk feature is the last non-task.)

- **Good alternates:**
  - `avgpool` → `[1, 2048, 1, 1]`
  - Spatial: `layer4.2.relu_2` → `[1, 2048, 7, 7]`

---

### vicreg_resnet50

- **Pick:** `flatten` → `[1, 2048]`  
  (VICReg uses a projector during pretraining; for general features you want the trunk pooled vector.)

- **Good alternates:**
  - `avgpool` → `[1, 2048, 1, 1]`
  - Spatial: `layer4.2.last_activation` → `[1, 2048, 7, 7]`

---

### vissl_resnet50_barlowtwins *(your run errored)*

- **Pick (conceptual):** pooled trunk feature right before `fc`  
  (i.e., `avgpool`→`flatten` → 2048-d penultimate.)

- **Alternate:** last conv map from `layer4.2` if you need spatial.

---

## ViT-L (SLIP variants)

**Why this choice:**  
For ViTs (timm-style), the task head consumes either the CLS token (after final norm) or a pooled token representation. In your traces, the sequence is:  
`... → norm → getitem_1 (CLS) → fc_norm → head_drop → head`  
So `fc_norm` (often called “pre-logits”) is the closest non-task tensor. `head` is the task head (classification/contrastive); we stop just before it.

### slip_vit_l_slip

- **Pick:** `fc_norm` → `[1, 1024]`  
  (Closest pre-head embedding; stable, task-agnostic.)

- **Good alternates:**
  - `getitem_1` (CLS after final norm) → `[1, 1024]`
  - Mean-pooled patch tokens (exclude CLS): take `norm.layer_norm` output `[1, 197, 1024]` and average over tokens `[:, 1:, :]` → `[1, 1024]`. This can outperform CLS in some retrieval/transfer or if you want less CLS idiosyncrasy.
  - Token-level features: keep `norm.layer_norm` `[1, 197, 1024]` if you’ll pool downstream (GeM/attention).

---

### slip_vit_l_simclr

- **Pick:** `fc_norm` → `[1, 1024]`

- **Good alternates:**
  - `getitem_1` (CLS after final norm) → `[1, 1024]`
  - Mean-pooled patch tokens (exclude CLS), as above.

---

### dinov2_vitl14

- **Primary (recommended):** `head` → shape `[B, 1024]`  
  *In DINOv2 the exposed head is the default image embedding used downstream. It’s not a classifier; it’s the projection of the final representation that the authors intend you to use.*

- **Good alternatives:**
  - `norm` → take CLS token after final LN: `norm[:, 0]` → `[B, 1024]`. Slightly closer to the trunk, sometimes preferred for analyses that want to avoid any projection.
  - `norm` → mean-pool patch tokens (exclude CLS): `norm[:, 1:].mean(dim=1)` → `[B, 1024]`. Often strong for retrieval/linear eval.

- **When to pick which:**
  - Use `head` for “what DINOv2 exposes as the embedding.”
  - Use `norm[:, 0]` for strict “pre-head” analyses.
  - Use mean-pooled `norm[:, 1:]` when you want spatially aggregated info rather than pure CLS.

---

### openclip_vit_so400m_14_siglip_webli

- **Primary (recommended):** `trunk.fc_norm` (the post-pool, pre-head feature) → `[B, 1152]`  
  *This is the representation right after the visual trunk’s pooling and normalization, but before the vision projection head that’s tuned for image-text alignment. It’s the safest “last non-task” feature.*

- **Good alternatives:**
  - `trunk.head` → `[B, 1152]`. If the head is Identity (common in OpenCLIP configs), this equals `fc_norm`. If it’s a learned linear, then it’s one small step more “text-aligned.”
  - `trunk.attn_pool.getitem_5` → `[B, 1152]`. Raw pooled token before the final LN; slightly earlier than `fc_norm`.

- **When to pick which:**
  - Prefer `trunk.fc_norm` for general downstream tasks and encoding models.
  - If you’re reproducing CLIP/SigLIP-style zero-shot behavior, `trunk.head` matches the model’s exported vision embedding.

---

### timm_vit_large_patch14_clip_224_laion2b

- **Primary (recommended):** `fc_norm` → `[B, 1024]`
  *This is the CLS embedding after the final norm but before the linear head; it stays model-agnostic while keeping the full ViT trunk context.*

- **Good alternatives:**
  - `getitem_1` (CLS token right after `norm`) → `[B, 1024]` if you want the raw CLS without the extra projection block.
  - Mean-pool `norm[:, 1:]` → `[B, 1024]` when you prefer pooled patch tokens over CLS.
  - `head` → `[B, 1000]` when you need the classification head activations.

---

### timm_vit_large_patch14_clip_quickgelu_224_dfn2b

- **Primary (recommended):** `fc_norm` → `[B, 1024]`
  *Identical ViT trunk ordering as the laion2b variant; stop right before the classifier to avoid label bias.*

- **Good alternatives:**
  - `getitem_1` (CLS after `norm`) → `[B, 1024]` for the immediate pre-`fc_norm` tensor.
  - Mean-pool `norm[:, 1:]` → `[B, 1024]` to aggregate patch tokens.
  - `head` → `[B, 1000]` when reproducing the provided logits.

---

### timm_vit_large_patch14_clip_quickgelu_224_openai

- **Primary (recommended):** `fc_norm` → `[B, 1024]`

- **Good alternatives:**
  - `getitem_1` (CLS post-`norm`) → `[B, 1024]`.
  - Mean-pool `norm[:, 1:]` → `[B, 1024]` for patch-token averaging.
  - `head` → `[B, 1000]` if you need the classifier output block.

---

### openclip_vit_l_14_quickgelu_metaclip_400m

- **Primary (recommended):** `ln_post[:, 0]` → `[B, 1024]`
  *This retains the CLS token after the final LayerNorm but before the projection head that maps to the shared image-text space.*

- **Good alternatives:**
  - Mean-pool `ln_post[:, 1:]` → `[B, 1024]` to blend patch tokens and reduce CLS variance.
  - `proj @ ln_post[:, 0]` (`matmul`) → `[B, 768]` when you want the exported CLIP vision embedding aligned to text.

---

### openclip_vit_l_14_quickgelu_metaclip_fullcc

- **Primary (recommended):** `ln_post[:, 0]` → `[B, 1024]`

- **Good alternatives:**
  - Mean-pool `ln_post[:, 1:]` → `[B, 1024]` for patch aggregation.
  - `proj` output → `[B, 768]` to stay aligned with CLIP zero-shot embeddings.

---

### openclip_vit_l_14_laion400m_e31

- **Primary (recommended):** `ln_post[:, 0]` → `[B, 1024]`

- **Good alternatives:**
  - Mean-pool `ln_post[:, 1:]` → `[B, 1024]`.
  - `proj` output → `[B, 768]` when you need the released OpenCLIP embedding.

---

### torchvision_vit_l_16_imagenet1k_v1

- **Primary (recommended):** `encoder.ln[:, 0]` → `[B, 1024]`
  *Take the CLS token after the final encoder LayerNorm; it is the penultimate representation before the classifier head.*

- **Good alternatives:**
  - Mean-pool `encoder.ln[:, 1:]` → `[B, 1024]` for patch-token aggregation.
  - Keep `encoder.ln` → `[B, 197, 1024]` when you plan to pool tokens downstream.
  - `heads.head` → `[B, 1000]` if you explicitly need logits.

---

## torchvision_vgg16_bn_imagenet1k_v1

- **Primary (recommended):** `classifier.5` (ReLU output before final `classifier.6`) → `[B, 4096]`  
  *That’s the penultimate activation right before the 1000-way classifier.*

- **Good alternatives:**
  - `classifier.2` (first 4096 ReLU) → `[B, 4096]` if you want a slightly earlier FC feature.
  - `avgpool+flatten` → `[B, 25088]` if you want conv features without the FC bottleneck (heavier, but sometimes better for linear evals).

- **Note:** `vgg16_imagenet21k` follows the same head layout; use the identical picks above.

---

## torchvision_resnet50_imagenet1k_v1

- **Primary (recommended):** `flatten` (i.e., output of global avgpool) → `[B, 2048]`  
  *This is the standard penultimate ResNet feature used everywhere; it sits right before `fc`.*

- **Good alternatives:**
  - `avgpool` (then squeeze/flatten) → `[B, 2048]` (identical numerically to flatten, just before the view).
  - `layer4` output (spatial map `[B, 2048, 7, 7]`), if you need spatial features; then pool yourself.

- **Note:** `resnet50_imagenet21k` and `robustness_imagenet_l2_eps3` share the same ResNet-50 trunk; stick with these layers (`model.flatten` / `model.avgpool` for the robustness model).

---

## torchvision_convnext_base_imagenet1k_v1

- **Primary (recommended):** global-pooled feature right before the classifier linear, i.e.:
  - Take `features` output `[B, 1024, 7, 7]`, then mean over H,W → `[B, 1024]`, then (optionally) take `classifier[0]` LayerNorm to stay pre-head.
  - Concretely: global mean pool (`features`) → `classifier[0]` (LayerNorm) → `[B, 1024]`.

- **Good alternatives:**
  - Just the global mean pool of features (skip LN) → `[B, 1024]` if you want the pure backbone output.

- **Avoid:** `classifier[2]` (that’s the 1000-way head).

---

## torchvision_alexnet_imagenet1k_v1 / alexnet_imagenet21k

- **Primary (recommended):** `classifier.5` → `[B, 4096]`
  *This is the final ReLU activation before the classifier linear; grab it for task-agnostic features.*

- **Good alternatives:**
  - `classifier.2` → `[B, 4096]` if you prefer the first FC ReLU.
  - `flatten` → `[B, 9216]` when you want conv features prior to the FC stack.

---

## cornet_s

- **Primary (recommended):** `decoder.avgpool` → squeeze to `[B, 512]`
  *This is the globally pooled IT representation immediately before the linear decoder; take it to stay pre-classifier.*

- **Good alternatives:**
  - `IT` block output → `[B, 512, 14, 14]` if you need spatial IT features.
  - `decoder.linear` → `[B, num_classes]` only when logits are required.

---