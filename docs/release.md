# Release Runbook

Do not run the remote publication steps until the repository owner has reviewed
the GPU outputs, artifact checksums, and final diff.

## 1. Local Preflight

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m pip check
pip-audit --requirement requirements.txt --progress-spinner off
python -m build
twine check dist/*
```

## 2. GPU Gate

Run on a CUDA host with access to all three model repositories:

```bash
TPSO_RUN_GPU_TESTS=1 pytest -q -m gpu tests/integration/test_gpu_generation.py
```

Inspect the four generated images from each model. This gate must not be replaced
by a CPU test, mock pipeline, or CLI dry run.

## 3. Build Context Artifacts

```bash
tpso-precompute \
  --models sd15 sd21 sd35 \
  --output-dir artifacts/unconditional

sha256sum artifacts/unconditional/*.pt
```

Copy the three generated hashes into `src/tpso/data/contexts.json`, then rerun
the local preflight. The packaged manifest must contain no `null` checksum.

## 4. Publish Contexts To Hugging Face

The following commands are intentionally manual:

```bash
hf auth login
hf repos create PonyMeng/TPSO --repo-type model --exist-ok

hf upload PonyMeng/TPSO \
  huggingface/README.md README.md --repo-type model
hf upload PonyMeng/TPSO \
  artifacts/unconditional/sd15_kappa0.8_lambda1.pt \
  sd15_kappa0.8_lambda1.pt --repo-type model
hf upload PonyMeng/TPSO \
  artifacts/unconditional/sd21_kappa0.8_lambda1.pt \
  sd21_kappa0.8_lambda1.pt --repo-type model
hf upload PonyMeng/TPSO \
  artifacts/unconditional/sd35_kappa0.8_lambda0.pt \
  sd35_kappa0.8_lambda0.pt --repo-type model

hf repos tag create PonyMeng/TPSO v0.1.0 \
  --repo-type model \
  --message "TPSO v0.1.0 unconditional contexts"
```

After upload, prove that manifest-driven download and checksum validation work
from a clean temporary cache without deleting any local artifact:

```bash
python -c "from tpso.contexts import download_context; print(download_context('sd15', cache_dir='/tmp/tpso-context-test'))"
python -c "from tpso.contexts import download_context; print(download_context('sd21', cache_dir='/tmp/tpso-context-test'))"
python -c "from tpso.contexts import download_context; print(download_context('sd35', cache_dir='/tmp/tpso-context-test'))"
```

## 5. Prepare The GitHub Release Worktree

The target GitHub repository may already contain an initial README or license.
Clone it into a separate sibling directory rather than forcing unrelated local
history onto `main`:

```bash
cd ..
git clone git@github.com:Open-Debin/TPSO.git TPSO-publish
rsync -av \
  --exclude=.git \
  --exclude=build \
  --exclude=dist \
  --exclude='*.egg-info' \
  --exclude='__pycache__' \
  --exclude=.pytest_cache \
  --exclude=.ruff_cache \
  --exclude=artifacts/unconditional \
  TPSO/ TPSO-publish/
cd TPSO-publish
```

Repeat the local preflight in `TPSO-publish` and inspect `git diff` before any
commit.

## 6. Commit, Tag, And Push

Run only after explicit owner confirmation:

```bash
git add --all
git commit -m "release: TPSO v0.1.0"
git tag -a v0.1.0 -m "TPSO v0.1.0"
git push origin main
git push origin v0.1.0
```

No model weights, generated images, paper PDFs, datasets, experiment outputs,
or cluster-specific files should appear in `git status` before publication.
