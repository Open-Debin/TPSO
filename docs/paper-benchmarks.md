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

Use `--dry-run` to inspect experiments and `--experiment NAME` to run one row.
Do not reuse an output directory with different settings because its
`manifest.json` must match the requested protocol.
