---
library_name: diffusers
tags:
  - text-to-image
  - stable-diffusion
  - prompt-optimization
  - tpso
---

# TPSO Unconditional Contexts

Precomputed unconditional contexts for
[TPSO](https://github.com/Open-Debin/TPSO) v0.1.0. This repository does not
contain diffusion-model weights.

| Model | File |
| --- | --- |
| Stable Diffusion 1.5 | `sd15_kappa0.8_lambda1.pt` |
| Stable Diffusion 2.1 | `sd21_kappa0.8_lambda1.pt` |
| Stable Diffusion 3.5 Medium | `sd35_kappa0.8_lambda0.pt` |

TPSO downloads and verifies the matching file automatically:

```bash
tpso-generate \
  --model sd15 \
  --prompt "A photograph of a red panda in a bamboo forest" \
  --num-images 4
```

Each file contains optimized CLIP representations and metadata. The TPSO
loader uses `torch.load(..., weights_only=True)` and verifies its SHA-256 digest.

TPSO code is Apache-2.0. Upstream model licenses and access terms still apply.
