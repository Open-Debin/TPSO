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

Image generation requires Python 3.10 or newer and a CUDA-capable GPU.

```bash
git clone https://github.com/Open-Debin/TPSO.git
cd TPSO
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Conda installation:

```bash
conda env create -f environment.yml
conda activate tpso
python -m pip install -e . --no-deps
```

Accept the required model licenses on Hugging Face, then authenticate if the
selected model is gated:

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

`--model` accepts `sd15`, `sd21`, or `sd35`. `--num-images 4` produces four
variants of this prompt. They are saved in `outputs/red-panda` as:

```text
outputs/red-panda/
|-- 0_0.jpg
|-- 0_1.jpg
|-- 0_2.jpg
`-- 0_3.jpg
```

The first number is the prompt index and the second is the variant index. To
generate images for several prompts, repeat `--prompt` once for each prompt:

```bash
tpso-generate \
  --model sd15 \
  --prompt "A red panda in a bamboo forest" \
  --prompt "A wooden chair beside a window" \
  --num-images 4 \
  --output-dir outputs/two-prompts
```

The first prompt is saved as `0_0.jpg` through `0_3.jpg`. The second prompt is
saved as `1_0.jpg` through `1_3.jpg`. This is the same naming convention used
by the paper benchmarks.

The model and precomputed unconditional context are downloaded on first use and
then loaded from the local cache.

## Configuration

The conditional embedding is optimized for each prompt. The unconditional
embedding is prompt-independent, so TPSO downloads a precomputed version from
[PonyMeng/TPSO](https://huggingface.co/PonyMeng/TPSO) and reuses it.

| Option | Meaning |
| --- | --- |
| `--config PATH` | Load a YAML configuration from `configs/`. |
| `--kappa FLOAT` | Set the target semantic similarity `kappa` (Table II). |
| `--diversity-weight FLOAT` | Set the diversity-loss weight `lambda` (Table V). |
| `--scheduler-ratio FLOAT` | Set the scheduling ratio `r`; a negative value reverses its direction (Table III). |
| `--offset-init NAME` | Select the token-offset initialization distribution (Table IV). |
| `--overwrite` | Replace images that already exist in the output directory. |

Options written on the command line override values loaded from `--config`.

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
the experiment. If any parameter changes, use a new `--output-root` so results
from different settings are not mixed.

To generate the complete Table I benchmark, omit `--experiment` and
`--limit-prompts`:

```bash
tpso-benchmark --group table1 --output-root outputs/paper
```

This runs SD1.5, SD2.1, and SD3.5. Each model uses 5,000 prompts and generates
10 variants per prompt, resulting in 150,000 images across the three models.

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
