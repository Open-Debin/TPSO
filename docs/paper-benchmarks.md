# Paper Benchmark Generation

`tpso-benchmark` reproduces the image-generation protocols for Tables I-V of
the official eight-page IJCNN 2026 paper. It does not compute metrics; metric
evaluation can be run after all expected images have been generated.

## Dataset

The archived TPSO code reads the public
`coco_30k_randomly_sampled_2014_val.csv` file maintained in the
`sayakpaul/sample-datasets` Hugging Face dataset. The benchmark downloads the
same file and accepts it only when its SHA-256 is:

```text
4e34947cb2a5d77c9bbaa11e8032e1339a26ebc32d9535725ab4f8e36760b7d3
```

Table I uses the first 5,000 caption rows and generates 10 variants per caption
(50,000 images per model). Tables II-V use the first 1,000 rows and generate 10
variants per caption (10,000 images per row). Pass the downloaded file
explicitly on a compute node without network access:

```bash
--prompts-csv artifacts/datasets/coco_30k_randomly_sampled_2014_val.csv
```

## Protocol

Images use the archived `{prompt_id}_{variant_id}.jpg` convention. Every result
directory also contains `manifest.json` and `prompts.csv`. The manifest records
the exact model revision, TPSO configuration, caption checksums, context
checksum, seed, batch size, image count, and generation/save resolutions.

For numerical comparison with the published tables, the benchmark preserves
the archived image protocol where it differs from the paper's prose:

- SD1.5 generates and saves at 512 px with 50 denoising steps.
- SD2.1 generates at 768 px and saves a 512 px Lanczos-resized JPEG with 50 steps.
- SD3.5 generates and saves at 512 px with 35 steps and T5 sequence length 77.
- Ten variants of one caption share the same initial diffusion latent.
- Source-compatible batches contain 5 prompts for SD1.5/SD2.1 and 2 prompts
  for SD3.5. Changing this size changes the semantic/diversity gradient ratio.
- The benchmark base seed is `2024`, recovered from the archived source.
- Fixed batches use seed `base_seed + prompt_start`, so interrupted runs resume
  without changing completed or regenerated batch results.

The generic `tpso-generate` command uses the same source-faithful defaults.

Tables II-IV use `lambda=0`, matching the semantic-only ablation represented by
their published values. Table V uses the three paper rows `lambda=0,5,10`.
The same model-specific unconditional context is reused across these conditional
ablations; its backbone, model revisions, encoder shapes, and group size remain
strictly validated. The SD3.5 main configuration uses its semantic-only
unconditional context.

## Commands

Inspect every row without loading a model or downloading data:

```bash
tpso-benchmark --group all --dry-run
```

Run one table at a time from the repository root:

```bash
COMMON_ARGS="--prompts-csv artifacts/datasets/coco_30k_randomly_sampled_2014_val.csv --context-dir artifacts/unconditional --output-root outputs/paper-reproduction"

tpso-benchmark --group table1 $COMMON_ARGS
tpso-benchmark --group table2 $COMMON_ARGS
tpso-benchmark --group table3 $COMMON_ARGS
tpso-benchmark --group table4 $COMMON_ARGS
tpso-benchmark --group table5 $COMMON_ARGS
```

Run or resume one row:

```bash
tpso-benchmark \
  --group table3 \
  --experiment fine_to_coarse_rm0p4 \
  --prompts-csv artifacts/datasets/coco_30k_randomly_sampled_2014_val.csv \
  --context-dir artifacts/unconditional \
  --output-root outputs/paper-reproduction
```

The default batch size follows the archived runs: five prompts for SD1.5/2.1
and two prompts for SD3.5. This matters because the semantic loss is summed
while the diversity loss is averaged. These defaults are safe for the tested
46 GB GPU. Increase `--batch-size` only after a short GPU test. Use
`--limit-prompts 2` for a non-paper debug run; never mix limited and full runs
in the same output root because their manifests intentionally differ.

Completed fixed batches are skipped. A partially written batch is regenerated
with its original deterministic seed. TPSO refuses to continue if the existing
manifest differs from the requested protocol.
