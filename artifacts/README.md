# Generated Artifacts

This directory is reserved for locally generated unconditional TPSO contexts.
Tensor files are excluded from Git and will be published separately through
Hugging Face Hub.

## Release Mapping

| Model | Local and Hub filename | Hub repository | SHA-256 |
| --- | --- | --- | --- |
| SD1.5 | `sd15_kappa0.8_lambda1.pt` | `Open-Debin/TPSO` | `8e0c7c1f...aadb5e` |
| SD2.1 | `sd21_kappa0.8_lambda1.pt` | `Open-Debin/TPSO` | `a97e85151d...2f925` |
| SD3.5 | `sd35_kappa0.8_lambda0.pt` | `Open-Debin/TPSO` | `7fcba1c236...452ba` |

Generate all three files and a candidate manifest with:

```bash
tpso-precompute \
  --models sd15 sd21 sd35 \
  --output-dir artifacts/unconditional
```

After uploading the tensor files, copy the generated SHA-256 values into
`src/tpso/data/contexts.json`. Automatic downloads remain disabled for an
artifact whose checksum is `null`.

Checkpoints use schema format version 1 and artifact version 0.1.0. They contain
only `metadata` plus model-specific CLIP encoder tensor mappings. Public loading
uses PyTorch's restricted `weights_only=True` mode and verifies downloaded files
against the manifest.
