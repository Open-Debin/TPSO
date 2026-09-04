# TPSO

Official implementation of **TPSO: Training-Free Diverse Image Generation via
Semantic Prompt Embedding Optimization** (IJCNN 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2511.19811-b31b1b.svg)](https://arxiv.org/abs/2511.19811)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

TPSO improves text-to-image diversity without training or modifying diffusion
model weights. It optimizes CLIP token embeddings under semantic and diversity
constraints, then progressively returns to the original condition during
denoising.

<p align="center">
  <img src="assets/comparison_baselines.png" alt="TPSO qualitative comparison" width="100%">
</p>

<p align="center">
  <img src="assets/pipeline_optim_infer.png" alt="TPSO pipeline" width="100%">
</p>

## Supported Models

| Name | Backbone | Resolution | Default lambda |
| --- | --- | ---: | ---: |
| `sd15` | Stable Diffusion 1.5 | 512 | 1 |
| `sd21` | Stable Diffusion 2.1 | 768 | 1 |
| `sd35` | Stable Diffusion 3.5 Medium | 512 | 10 |

For SD3.5, TPSO optimizes both CLIP encoders and keeps the T5 representation
unchanged.

## Installation

Python 3.10+ and a CUDA-capable GPU are required. Install the PyTorch build for
your CUDA platform, then install TPSO:

```bash
git clone https://github.com/Open-Debin/TPSO.git
cd TPSO
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Conda users can instead run:

```bash
conda env create -f environment.yml
conda activate tpso
python -m pip install -e . --no-deps
```

Authenticate when a gated model requires access:

```bash
hf auth login
```

## Quick Start

```bash
tpso-generate \
  --model sd15 \
  --prompt "A photograph of a red panda in a bamboo forest" \
  --num-images 4 \
  --output-dir outputs/red-panda
```

Choose `sd15`, `sd21`, or `sd35` with `--model`. The matching precomputed
unconditional context is downloaded automatically from
[PonyMeng/TPSO](https://huggingface.co/PonyMeng/TPSO).

Use a configuration file or override parameters directly:

```bash
tpso-generate \
  --config configs/sd15.yaml \
  --prompt "A wooden chair" \
  --kappa 0.8 \
  --output-dir outputs/chair
```

Rebuild the unconditional context locally when needed:

```bash
tpso-generate \
  --model sd15 \
  --prompt "A wooden chair" \
  --rebuild-unconditional \
  --output-dir outputs/chair-rebuilt
```

## Generate From 1,000 Prompts

The CSV must contain `file_name` and `caption` columns. This command uses its
first 1,000 prompts and generates 10 images per prompt:

```bash
tpso-benchmark \
  --group table1 \
  --experiment sd15 \
  --limit-prompts 1000 \
  --prompts-csv path/to/coco_30k_randomly_sampled_2014_val.csv \
  --output-root outputs/coco-1k
```

Outputs use `{prompt_id}_{variant_id}.jpg`, producing 10,000 images. Replace
`sd15` with `sd21` or `sd35`. See [benchmark generation](docs/paper-benchmarks.md)
for the paper protocols.

## Citation

```bibtex
@inproceedings{meng2026tpso,
  title     = {TPSO: Training-Free Diverse Image Generation via Semantic Prompt Embedding Optimization},
  author    = {Meng, Debin and Jin, Chen and Gao, Zheng and Li, Yanran and Patras, Ioannis and Tzimiropoulos, Georgios},
  booktitle = {International Joint Conference on Neural Networks},
  year      = {2026}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). Model weights retain
their original licenses.
