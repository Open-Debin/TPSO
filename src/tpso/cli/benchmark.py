"""Command-line entry point for TPSO paper-generation protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpso.benchmark import (
    PAPER_VARIANTS,
    SOURCE_BATCH_SIZE,
    describe_experiment,
    load_coco_prompts,
    resolve_coco_csv,
    run_experiment,
    select_experiments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TPSO Tables I-V evaluation images.")
    parser.add_argument(
        "--group",
        required=True,
        choices=("all", "table1", "table2", "table3", "table4", "table5"),
    )
    parser.add_argument(
        "--experiment", action="append", help="Run only a named row; repeat as needed."
    )
    parser.add_argument("--output-root", default="outputs/paper-reproduction")
    parser.add_argument(
        "--prompts-csv", help="Exact archived MS-COCO caption CSV; downloaded if omitted."
    )
    parser.add_argument("--context-dir", default="artifacts/unconditional")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Prompts per batch; defaults to the archived model-specific value.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--limit-prompts",
        type=int,
        help="Debug only: truncate every selected row instead of running the paper sample count.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.limit_prompts is not None and args.limit_prompts <= 0:
        raise ValueError("--limit-prompts must be positive.")
    experiments = select_experiments(args.group, args.experiment)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "variants_per_prompt": PAPER_VARIANTS,
                    "output_root": str(Path(args.output_root)),
                    "experiments": [describe_experiment(item) for item in experiments],
                },
                indent=2,
            )
        )
        return 0

    csv_path = resolve_coco_csv(args.prompts_csv)
    for experiment in experiments:
        count = experiment.prompt_count
        if args.limit_prompts is not None:
            count = min(count, args.limit_prompts)
        records = load_coco_prompts(csv_path, count)
        output = run_experiment(
            experiment,
            records,
            output_root=args.output_root,
            batch_size=args.batch_size or SOURCE_BATCH_SIZE[experiment.model],
            seed=args.seed,
            device=args.device,
            precision=args.precision,
            context_dir=args.context_dir,
            local_files_only=args.local_files_only,
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
