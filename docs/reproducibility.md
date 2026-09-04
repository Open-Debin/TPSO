# Reproducibility

## Authoritative Paper

This release was checked against the official eight-page IJCNN 2026 PDF with
SHA-256
`196dfb5aaac629896b225a770b72ce9b74158b425ad00edc8bed08cd00841a2b`.
Other draft or supplementary PDFs are not used as implementation authority.
Where the official paper is silent, this document labels settings recovered
from the archived main-table run records rather than presenting them as stated
paper parameters.

This release follows the archived executable protocol when prose and code differ.
SD2.1 generates at 768 pixels; the benchmark saves a 512-pixel resize. SD3.5
generates at 512 pixels with 35 steps, T5 length 77, and stochastic scheduler
noise scale 0.03.

## Verified Without GPU

The regular test suite is hardware-independent and checks:

- paper-default configuration;
- semantic and pairwise diversity losses;
- the coarse-to-fine interpolation schedule;
- source-faithful RMSprop offset convergence on a differentiable toy encoder;
- context serialization, sampling, and SHA-256 manifests;
- package imports and CLI dry runs.

Published context checkpoints record the logical model, Hugging Face model ID,
projection-model ID, optimizer configuration, and sample count. A checkpoint for
another model mapping is rejected rather than silently reused. Context rows are
sampled without replacement when the checkpoint contains enough rows.

The paper does not state a mixed-precision mode, while the archived research
implementation uses FP16. `--precision auto` therefore resolves to FP16 on CUDA
and FP32 on CPU. The public CLI currently defaults to BF16 for improved SD3.5
numerical stability; pass `--precision fp16` for the archived precision path.

The release CPU suite was validated with Python 3.12.14, PyTorch 2.6.0,
Diffusers 0.38.0, Transformers 5.5.4, and Accelerate 0.30.0. Final CUDA
validation remains part of the GPU release gate below.

The provided Conda file uses Conda for Python and pip for PyTorch and TPSO.
Users needing a CUDA-specific PyTorch wheel should install the matching wheel
for their driver/platform before installing TPSO; an already installed
compatible PyTorch satisfies the package requirement.

## Model Revisions

| Component | Repository | Revision |
| --- | --- | --- |
| SD1.5 | `sd-legacy/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| SD1.5 projection | `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` |
| SD2.1 | `sd2-community/stable-diffusion-2-1` | `bb2154823665391b4fb29b0b9cf82a198964ee05` |
| SD2.1 projection | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | `1c2b8495b28150b8a4922ee1c8edee224c284c0c` |
| SD3.5 Medium | `stabilityai/stable-diffusion-3.5-medium` | `b940f670f0eda2d07fbb75229e779da1ad11eb80` |

SD2.1 uses a community mirror because the original Stability AI Hub repository
was deprecated and now returns 404. The mirror declares itself an unaffiliated
copy and retains the OpenRAIL++ model license. TPSO does not redistribute any
of these weights.

## GPU Release Gate

Before tagging `v0.1.0`, run one end-to-end generation for each of `sd15`,
`sd21`, and `sd35`, verify that all context artifacts pass their checksums, and
visually inspect at least four outputs per prompt. A CPU dry run does not satisfy
this gate.

Recommended commands:

```bash
tpso-generate --model sd15 --prompt "A wooden chair" --num-images 4
tpso-generate --model sd21 --prompt "A wooden chair" --num-images 4
tpso-generate --model sd35 --prompt "A wooden chair" --num-images 4
```

Record the GPU, driver, CUDA, PyTorch, Diffusers, and Transformers versions with
the release notes. Do not commit generated images or model weights.
