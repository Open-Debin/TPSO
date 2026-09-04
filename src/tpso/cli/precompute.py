"""Precompute reusable unconditional TPSO contexts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from tpso.config import MODEL_SPECS, default_config
from tpso.contexts import ARTIFACT_VERSION, FORMAT_VERSION, sha256_file
from tpso.runner import SOURCE_CONTEXT_PROTOCOL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute TPSO unconditional contexts.")
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS)
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/unconditional"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repo-id", default="Open-Debin/TPSO")
    parser.add_argument("--revision", default="v0.1.0")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing artifacts.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = [args.output_dir / MODEL_SPECS[model].context_filename for model in args.models]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "models": args.models,
                    "protocols": {
                        model: {
                            "count": SOURCE_CONTEXT_PROTOCOL[model]["count"],
                            "group_size": SOURCE_CONTEXT_PROTOCOL[model]["group_size"],
                        }
                        for model in args.models
                    },
                    "outputs": [str(path) for path in outputs],
                },
                indent=2,
            )
        )
        return 0
    manifest_path = args.output_dir / "contexts.json"
    conflicts = [path for path in [*outputs, manifest_path] if path.exists()]
    if conflicts and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(conflicts)} context artifact(s); "
            "choose another output directory or pass --overwrite."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    from tpso.runner import (
        load_adapter,
        rebuild_context,
        resolve_dtype,
        unconditional_config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tpso-precompute-", dir=args.output_dir) as staging_name:
        staging_dir = Path(staging_name)
        staged_outputs = []
        artifacts = []
        for index, model in enumerate(args.models):
            spec = MODEL_SPECS[model]
            dtype = resolve_dtype(args.device, args.precision)
            protocol = SOURCE_CONTEXT_PROTOCOL[model]
            adapter = load_adapter(
                spec,
                device=args.device,
                dtype=dtype,
                local_files_only=args.local_files_only,
            )
            path = rebuild_context(
                adapter,
                spec,
                unconditional_config(spec, default_config(model)),
                count=protocol["count"],
                group_size=protocol["group_size"],
                seed=args.seed + index,
                output_path=staging_dir / spec.context_filename,
            )
            staged_outputs.append(path)
            artifacts.append(
                {
                    "model": model,
                    "filename": spec.context_filename,
                    "repo_id": args.repo_id,
                    "revision": args.revision,
                    "sha256": sha256_file(path),
                }
            )
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        staged_manifest = staging_dir / manifest_path.name
        staged_manifest.write_text(
            json.dumps(
                {
                    "format_version": FORMAT_VERSION,
                    "artifact_version": ARTIFACT_VERSION,
                    "artifacts": artifacts,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for staged, destination in zip(staged_outputs, outputs, strict=True):
            staged.replace(destination)
        staged_manifest.replace(manifest_path)
    print(manifest_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
