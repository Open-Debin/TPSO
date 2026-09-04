# Repository Structure

TPSO uses a `src/` package layout so tests exercise the installed package rather
than accidentally importing files from the repository root.

## Intended Public Layout

```text
TPSO/
|-- src/tpso/
|   |-- contexts.py          # Context schemas, validation, and Hub downloads
|   |-- losses.py            # Semantic and diversity objectives
|   |-- optimization.py      # Prompt-offset optimization loop
|   |-- scheduling.py        # Coarse-to-fine interpolation schedule
|   |-- config.py            # Typed configuration loading and validation
|   |-- pipelines/
|   |   |-- stable_diffusion.py
|   |   `-- stable_diffusion3.py
|   `-- cli/
|       |-- generate.py      # Unified user-facing inference command
|       `-- precompute.py    # Exact unconditional-context generation
|-- scripts/                 # Thin reproducibility wrappers
|-- configs/                 # Validated release defaults for supported backbones
|-- huggingface/             # Hub model card for context artifacts
|-- tests/
|   |-- unit/                # CPU-only mathematical and schema tests
|   `-- integration/         # Model-loading and GPU smoke tests
|-- docs/                    # Method and release documentation
`-- artifacts/               # Generated local assets, ignored by Git
```

## Release Design Rules

1. Preserve paper behavior without exposing the research-directory structure.
2. Keep `tpso` as the only public method and package name.
3. Remove cluster paths, notebook state, debugger calls, and experiment-only flags.
4. Preserve third-party licenses and attribution.
5. Keep CPU unit tests independent of model downloads.
6. Require one GPU smoke test per supported backbone before release.

## Artifact Distribution

GitHub stores source code and small documentation assets. Precomputed
unconditional contexts are versioned on Hugging Face Hub with checksums and
downloaded through `huggingface_hub`. Local copies remain under
`artifacts/unconditional/` and are never committed to Git.
