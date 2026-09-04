# TPSO

Official implementation of **TPSO: Training-Free Diverse Image Generation via
Semantic Prompt Embedding Optimization** (IJCNN 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2511.19811-b31b1b.svg)](https://arxiv.org/abs/2511.19811)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

TPSO increases the diversity of text-to-image generation without training or
modifying diffusion-model weights. Before denoising, it optimizes small offsets
added to the CLIP token embeddings. A semantic constraint keeps each optimized
embedding close to the original prompt, while a diversity loss encourages the
variants of that prompt to differ from one another. During denoising, TPSO
gradually returns to the original prompt embedding to preserve image quality.

<p align="center">
  <img src="assets/comparison_baselines.png" alt="TPSO qualitative comparison" width="100%">
</p>

<p align="center">
  <img src="assets/pipeline_optim_infer.png" alt="TPSO pipeline" width="100%">
</p>

## Supported Models

| CLI name | Backbone | Generation resolution | Default lambda |
| --- | --- | ---: | ---: |
| `sd15` | Stable Diffusion 1.5 | 512 | 1 |
| `sd21` | Stable Diffusion 2.1 | 768 | 1 |
| `sd35` | Stable Diffusion 3.5 Medium | 512 | 10 |

`lambda` is the weight of the diversity loss. For SD3.5, TPSO optimizes the two
CLIP encoders independently and leaves the T5 representation unchanged.

## Installation

Image generation requires Python 3.10 or newer and a CUDA-capable GPU. The
commands below clone the repository, create an isolated Python environment,
install the dependencies listed in `requirements.txt`, and install the TPSO
commands into that environment.

Install a PyTorch build compatible with your CUDA platform first if the default
PyPI package is not suitable for your machine.

```bash
git clone https://github.com/Open-Debin/TPSO.git
cd TPSO
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If you prefer Conda, `environment.yml` creates the equivalent `tpso`
environment:

```bash
conda env create -f environment.yml
conda activate tpso
python -m pip install -e . --no-deps
```

Some Stable Diffusion repositories require you to accept their license on
Hugging Face. After receiving access, authenticate on the machine that will run
TPSO:

```bash
hf auth login
```

## Generate Images

The following command generates four SD1.5 images from one prompt:

```bash
tpso-generate \
  --model sd15 \
  --prompt "A photograph of a red panda in a bamboo forest" \
  --num-images 4 \
  --output-dir outputs/red-panda
```

Here, `--model` selects the backbone, `--prompt` supplies the text condition,
and `--num-images` controls how many diverse variants are produced. The images
are saved as `00.png`, `01.png`, and so on inside `outputs/red-panda`.

On the first run, TPSO downloads the selected diffusion model and its matching
precomputed unconditional context. Later runs reuse the local Hugging Face
cache. Choose `sd15`, `sd21`, or `sd35` with `--model`. Use a new output
directory for a new run, or pass `--overwrite` if existing images may be
replaced.

Each supported model also has a YAML configuration in `configs/`. The following
command starts from the SD1.5 configuration and changes only `kappa` to `0.8`.
An explicit command-line option takes precedence over the value in the YAML
file.

```bash
tpso-generate \
  --config configs/sd15.yaml \
  --prompt "A wooden chair" \
  --kappa 0.8 \
  --output-dir outputs/chair
```

## Unconditional Context

Classifier-free guidance uses both conditional and unconditional prompt
embeddings. The conditional embedding depends on the user's prompt and must be
optimized for each generation. The unconditional embedding does not depend on
the prompt, so TPSO optimizes it once and reuses it.

By default, the correct context is downloaded from
[PonyMeng/TPSO](https://huggingface.co/PonyMeng/TPSO) and its checksum is
verified automatically. Most users do not need to manage this file manually.
To reproduce the precomputation locally instead, use:

```bash
tpso-generate \
  --model sd15 \
  --prompt "A wooden chair" \
  --rebuild-unconditional \
  --output-dir outputs/chair-rebuilt
```

This local rebuilding step is slower than loading the published context, but it
does not change how the conditional prompt is optimized.

## Generate From 1,000 Prompts

The paper experiments use `coco_30k_randomly_sampled_2014_val.csv`. The file has
two columns:

```csv
file_name,caption
COCO_val2014_000000054123.jpg,A group of zebras grazing in the grass.
COCO_val2014_000000012897.jpg,a number of people standing around a large group of luggage bags
```

The referenced COCO images are not required. `caption` provides the generation
prompt, while `file_name` is retained as source metadata.

The following command reads the first 1,000 rows and generates 10 variants for
each prompt with SD1.5:

```bash
tpso-benchmark \
  --group table1 \
  --experiment sd15 \
  --limit-prompts 1000 \
  --prompts-csv /path/to/coco_30k_randomly_sampled_2014_val.csv \
  --output-root outputs/coco-1k
```

`--group table1` selects the main-comparison presets, and `--experiment sd15`
selects only the SD1.5 row from that group. `--limit-prompts 1000` restricts the
run to the first 1,000 captions. Replace `sd15` with `sd21` or `sd35` to use a
different backbone.

The result directory contains 10,000 images named
`{prompt_id}_{variant_id}.jpg`, together with `prompts.csv` and `manifest.json`.
The manifest records the settings used for that run. If generation is
interrupted, rerunning the same command skips completed batches and continues
the experiment. If you change the model, prompt count, seed, or batch size, use
a new `--output-root` so results from different settings are not mixed.

See [the method-to-code map](docs/method.md) for implementation details and
[paper benchmark generation](docs/paper-benchmarks.md) for Tables I-V.

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
