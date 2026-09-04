# Unconditional Contexts

Generated tensor files are excluded from Git and published at
[PonyMeng/TPSO](https://huggingface.co/PonyMeng/TPSO).

| Model | File | SHA-256 |
| --- | --- | --- |
| SD1.5 | `sd15_kappa0.8_lambda1.pt` | `8e0c7c1f...aadb5e` |
| SD2.1 | `sd21_kappa0.8_lambda1.pt` | `a97e85151d...2f925` |
| SD3.5 | `sd35_kappa0.8_lambda0.pt` | `7fcba1c236...452ba` |

Rebuild all release contexts:

```bash
tpso-precompute \
  --models sd15 sd21 sd35 \
  --output-dir artifacts/unconditional
```

Full checksums are stored in `src/tpso/data/contexts.json` and verified after
download.
