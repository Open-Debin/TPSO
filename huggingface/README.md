---
library_name: diffusers
tags:
  - text-to-image
  - stable-diffusion
  - prompt-optimization
  - tpso
---

# TPSO Unconditional Contexts

This repository distributes the precomputed unconditional prompt contexts used
by [TPSO](https://github.com/Open-Debin/TPSO) v0.1.0. It does not contain Stable
Diffusion, CLIP, or T5 model weights.

## Artifacts

| TPSO model | Filename | Upstream model |
| --- | --- | --- |
| SD1.5 | `sd15_kappa0.8_lambda1.pt` | `sd-legacy/stable-diffusion-v1-5` |
| SD2.1 | `sd21_kappa0.8_lambda1.pt` | `sd2-community/stable-diffusion-2-1` |
| SD3.5 Medium | `sd35_kappa0.8_lambda0.pt` | `stabilityai/stable-diffusion-3.5-medium` |

The packaged TPSO manifest pins this Hub repository at revision `v0.1.0` and
records the SHA-256 digest for every file. The loader rejects artifacts with a
missing or mismatched digest.

## Format

Each file is a PyTorch tensor-only checkpoint with schema format version 1 and
artifact version 0.1.0. It contains metadata plus optimized and original CLIP
hidden and projected representations. TPSO loads it with
`torch.load(..., weights_only=True)`, validates the model and encoder shapes,
and moves only sampled rows to the inference device.

## Usage

Install TPSO and run its unified command. The matching context is downloaded
automatically:

```bash
tpso-generate \
  --model sd15 \
  --prompt "A photograph of a red panda in a bamboo forest" \
  --num-images 4
```

See the GitHub repository for installation, source code, paper defaults, model
revisions, and local context rebuilding instructions.

## Terms

TPSO source code is Apache-2.0. The upstream model licenses and access terms
continue to apply; users must review them before downloading or using the
corresponding model weights. The SD2.1 artifact is tied to an unaffiliated
community mirror of the deprecated Stability AI SD2.1 repository. SD3.5 Medium
is gated under the Stability AI Community License, including its commercial-use
conditions and annual-revenue threshold.
