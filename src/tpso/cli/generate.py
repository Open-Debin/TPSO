"""Unified TPSO generation command."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from tpso.config import MODEL_SPECS, default_config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate diverse images with TPSO on SD1.5, SD2.1, or SD3.5."
    )
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--prompt", required=True, action="append", help="Repeat for multiple prompts."
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", help="Optional model-specific YAML configuration.")
    parser.add_argument("--num-images", type=int, default=4, help="Images generated per prompt.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--context-path", help="Local unconditional-context checkpoint.")
    parser.add_argument("--rebuild-unconditional", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--diversity-weight", type=float)
    parser.add_argument("--scheduler-ratio", type=float)
    parser.add_argument(
        "--offset-init",
        choices=(
            "approx_zero",
            "normal",
            "zero",
            "normal_1",
            "normal_2",
            "laplace_1",
            "laplace_1.5",
            "laplace_sqrt1.5",
            "gamma_1_1",
            "gamma_4_4",
            "gamma_4_sqrt4",
            "gamma_7_sqrt7",
            "uniform_0_1",
            "uniform_-1_1",
            "uniform_-3_3",
        ),
    )
    parser.add_argument("--offset-std", type=float)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output images.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print resolved paper defaults without loading models.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config:
        spec, config = load_config(args.config, args.model)
    else:
        spec, config = MODEL_SPECS[args.model], default_config(args.model)
    config = replace(
        config,
        **{
            key: value
            for key, value in {
                "kappa": args.kappa,
                "diversity_weight": args.diversity_weight,
                "scheduler_ratio": args.scheduler_ratio,
                "offset_init": args.offset_init,
                "offset_std": args.offset_std,
            }.items()
            if value is not None
        },
    ).validate()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive.")
    if any(not prompt.strip() for prompt in args.prompt):
        raise ValueError("--prompt must contain at least one non-whitespace character.")
    from tpso.runner import resolve_generation_settings

    generation = resolve_generation_settings(
        spec,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": asdict(spec),
                    "tpso": asdict(config),
                    "prompts": args.prompt,
                    "num_images": args.num_images,
                    "context": args.context_path or "auto",
                    "generation": generation,
                },
                indent=2,
            )
        )
        return 0

    from tpso.runner import generate

    paths = generate(
        model=args.model,
        prompts=args.prompt,
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        context_path=args.context_path,
        rebuild_unconditional=args.rebuild_unconditional,
        local_files_only=args.local_files_only,
        config_path=args.config,
        kappa=args.kappa,
        diversity_weight=args.diversity_weight,
        scheduler_ratio=args.scheduler_ratio,
        offset_init=args.offset_init,
        offset_std=args.offset_std,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        overwrite=args.overwrite,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
