# Paper Benchmark Generation

`tpso-benchmark` generates images for Tables I-V. It does not compute metrics.

The input CSV must be the archived
`coco_30k_randomly_sampled_2014_val.csv` with `file_name` and `caption` columns.
If `--prompts-csv` is omitted, TPSO downloads and verifies it automatically.

```csv
file_name,caption
COCO_val2014_000000054123.jpg,A group of zebras grazing in the grass.
COCO_val2014_000000012897.jpg,a number of people standing around a large group of luggage bags
```

Only the captions are used for generation; the referenced COCO images are not
required.

## Generate 1,000 Prompts

```bash
tpso-benchmark \
  --group table1 \
  --experiment sd15 \
  --limit-prompts 1000 \
  --prompts-csv /path/to/coco_30k_randomly_sampled_2014_val.csv \
  --output-root outputs/coco-1k
```

This generates 10 variants per prompt as `{prompt_id}_{variant_id}.jpg`. Use
`sd21` or `sd35` for another backbone.

## Reproduce Paper Tables

```bash
tpso-benchmark --group table1 --output-root outputs/paper
tpso-benchmark --group table2 --output-root outputs/paper
tpso-benchmark --group table3 --output-root outputs/paper
tpso-benchmark --group table4 --output-root outputs/paper
tpso-benchmark --group table5 --output-root outputs/paper
```

Table I uses 5,000 prompts. Tables II-V use 1,000 prompts. Each prompt produces
10 images. Existing complete batches are skipped when resuming.

`--group table1` selects all three main-comparison experiments: SD1.5, SD2.1,
and SD3.5. Before generating images, you can inspect these selected experiments
with `--dry-run`:

```bash
tpso-benchmark --group table1 --dry-run
```

This prints the resolved settings and exits without loading a model or writing
images. To run only one experiment from Table I, add its name. For example:

```bash
tpso-benchmark \
  --group table1 \
  --experiment sd15 \
  --output-root outputs/paper
```

The result directory contains a `manifest.json` recording the model, prompt
count, seed, batch size, and other generation settings. If a run is interrupted,
execute the same command again: TPSO reads this manifest, skips complete
batches, and continues from the remaining prompts.

Do not write a different configuration into that same result directory. For
example, after running with `--batch-size 5`, a new run with `--batch-size 8`
should use a different root such as `outputs/paper-batch8`. This keeps images
generated with different settings separate and prevents accidental mixing.
