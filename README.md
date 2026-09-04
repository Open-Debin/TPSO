# TPSO

Official implementation of **TPSO: Training-Free Diverse Image Generation via Semantic Prompt Embedding Optimization** (IJCNN 2026).

TPSO improves text-to-image diversity without training or changing diffusion-model weights. It optimizes small offsets at the CLIP token-embedding level, constrains the resulting prompt embeddings to retain the original semantics, and progressively returns to the original condition during denoising.

## Release Status

This repository is preparing the `v0.1.0` release candidate. The package, CPU
tests, command-line dry runs, distribution metadata, and end-to-end GPU
generation on all three backbones are verified. Publication of the three
checksum-pinned unconditional-context artifacts remains pending.

## Supported Models

| CLI name | Backbone | Resolution | TPSO diversity weight |
| --- | --- | ---: | ---: |
| `sd15` | Stable Diffusion 1.5 | 512 | 1 |
| `sd21` | Stable Diffusion 2.1 | 768 | 1 |
| `sd35` | Stable Diffusion 3.5 Medium | 512 | 10 |

SD3.5 optimizes its two CLIP encoders independently. Its T5 representation is retained as the original condition.

The original `stabilityai/stable-diffusion-2-1` Hub repository was deprecated
and is no longer downloadable. TPSO therefore pins the public
`sd2-community/stable-diffusion-2-1` mirror at a fixed commit. The mirror is not
affiliated with Stability AI; its model card identifies it as a copy of the
deprecated SD2.1 repository. All other model and projection repositories are
also commit-pinned for reproducibility.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended.

Install the PyTorch wheel appropriate for your CUDA platform first when the
default PyPI wheel is not suitable; see the official PyTorch installation
selector. TPSO's release validation uses PyTorch 2.6.

```bash
git clone https://github.com/Open-Debin/TPSO.git
cd TPSO
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0
python -m pip install -e .
```

Alternatively, create the provided Conda environment from the repository root
so its local `requirements.txt` reference resolves correctly:

```bash
conda env create --file environment.yml
conda activate tpso
python -m pip install -e . --no-deps
```

Some model repositories require accepting their terms and authenticating with Hugging Face:

```bash
hf auth login
```

## Quick Start

One command supports all three backbones:

```bash
python scripts/generate.py \
  --model sd15 \
  --prompt "A photograph of a red panda in a bamboo forest" \
  --num-images 4 \
  --output-dir outputs/red-panda
```

Use `--model sd21` or `--model sd35` for the other paper backbones. Repeat `--prompt` to process several prompts in one run. The equivalent installed command is `tpso-generate`.

TPSO refuses to replace existing images in the output directory. Choose a new
directory for each run or pass `--overwrite` explicitly.

The validated YAML files in `configs/` are executable release configuration,
not documentation-only examples. Pass one with `--config configs/sd15.yaml`;
explicit CLI options such as `--kappa` override the YAML value.

Validate configuration without loading model weights:

```bash
tpso-generate --model sd35 --prompt "A field of flowers with blue sky" --dry-run
```

## Unconditional Contexts

TPSO optimizes both conditional and unconditional CLIP contexts. The unconditional prompt does not depend on the user's text, so its optimized contexts can be precomputed and reused. Rebuilding follows the archived experiment protocols: a single-space prompt with the first 20 interior CLIP positions enabled, using 350 contexts in groups of 35 for SD1.5/2.1 and 300 contexts in groups of 30 for SD3.5. SD3.5 uses the archived semantic-only unconditional context even when its conditional diversity weight is 10. These settings are recorded in every checkpoint's metadata.

Resolution order during generation:

1. Load `--context-path` when provided.
2. Otherwise download the model-specific artifact registered in the packaged manifest.
3. If no published artifact is available, rebuild and cache it locally.

Force an exact local rebuild with:

```bash
tpso-generate \
  --model sd15 \
  --prompt "A wooden chair" \
  --rebuild-unconditional
```

Precompute all release artifacts and a SHA-256 manifest:

```bash
tpso-precompute --models sd15 sd21 sd35 --output-dir artifacts/unconditional
```

Tensor artifacts are intentionally excluded from Git and are distributed separately through Hugging Face Hub. Until their checksums are inserted into `src/tpso/data/contexts.json`, automatic download refuses unverified artifacts and falls back to local rebuilding.

Published unconditional contexts use the paper-default precomputation protocol
and are reusable across conditional `kappa`, diversity-weight, and scheduler
settings. Checkpoints are still validated against the requested backbone, pinned
model revisions, encoder shapes, and context group size. Pass `--context-path`
to select a local checkpoint explicitly or `--rebuild-unconditional` to create a
fresh one.

## Paper Defaults

The defaults reproduce the archived code paths that generated the reported
results. Where the paper prose and executable source differ, the source-faithful
setting is identified below:

| Parameter | Value |
| --- | ---: |
| Semantic retention `kappa` | 0.8 |
| Convergence tolerance | 0.01 |
| Offset initialization | `Normal(0, 1e-4)` |
| Optimizer | RMSprop |
| Learning rate | 0.01 |
| Maximum optimization steps | 50 (SD1.5/2.1), 200 (SD3.5) |
| Coarse-to-fine ratio | 0.4 |
| Diversity weight | 1, or 10 for SD3.5 |

The official paper does not enumerate a separate diversity weight for each
backbone. The values above are release-reproduction settings: the archived
Table I result rows match the SD1.5 and SD2.1 runs with `lambda=1` and the
SD3.5 run with `lambda=10`. Table V in the paper separately reports the
SD1.5 ablation at `lambda=0`, `5`, and `10`.

The paper pseudocode names Adam, but the archived main-generation entry points
used RMSprop. This release defaults to RMSprop for numerical compatibility;
`optimizer: adam` remains available in a YAML config for studying the literal
pseudocode variant.

## Paper Benchmarks

The optional `tpso-benchmark` command generates the exact MS-COCO prompt subsets
and `{prompt_id}_{variant_id}.jpg` layout needed to evaluate Tables I-V. It
supports deterministic resumption and writes a complete protocol manifest into
every result directory:

```bash
tpso-benchmark --group table1 --dry-run
```

See [Paper Benchmark Generation](docs/paper-benchmarks.md) before starting the
full 330,000-image suite. Generated datasets remain excluded from Git.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
pip-audit --requirement requirements.txt --progress-spinner off
python -m build
twine check dist/*
```

CPU tests cover the losses, optimizer, scheduler, context schema, imports, and CLI. Full model generation requires GPU memory and access to the corresponding model weights.

Opt in to the model-download and GPU smoke suite with:

```bash
TPSO_RUN_GPU_TESTS=1 pytest -q -m gpu tests/integration/test_gpu_generation.py
```

See the [release runbook](docs/release.md) for the artifact, Hugging Face, and
GitHub publication sequence.

## Repository Scope

Version `0.1.0` contains only the TPSO inference method, three paper backbones, context tooling, tests, and documentation. Baseline methods, paper evaluation scripts, experiment data, notebooks, checkpoints, and generated images are deliberately excluded.

## Roadmap

- `v0.1.0`: TPSO inference for SD1.5, SD2.1, and SD3.5 with reusable contexts.
- Later releases: paper evaluation recipes and optional baseline integrations,
  kept separate from the core inference package.

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

The source code is released under the [Apache License 2.0](LICENSE). Stable Diffusion weights are not included and remain governed by their respective model licenses.
