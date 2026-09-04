# Method-to-Code Map

This document maps the official TPSO paper to the public implementation.

| Paper component | Public implementation |
| --- | --- |
| Token offsets, Equations (1)-(4) | `tpso.text_encoder.CLIPOffsetSession` |
| Semantic loss, Equation (5) | `tpso.losses.semantic_loss` |
| Diversity loss, Equation (6) | `tpso.losses.diversity_loss` |
| Joint objective, Equation (7) | `tpso.optimization.optimize_prompt_offsets` |
| Convergence, Equation (8) | `TPSOConfig.tolerance` and the active mask |
| Scheduler, Equations (9)-(10) | `tpso.scheduling` |
| SD1.5 and SD2.1 sampling | `tpso.pipelines.stable_diffusion` |
| SD3.5 dual-CLIP sampling | `tpso.pipelines.stable_diffusion3` |

SD1.5/2.1 use the archived whitespace word count, while SD3.5 uses the CLIP
end-of-text position. TPSO applies the resulting token mask both when offsets
are initialized and to their gradients. Non-semantic positions therefore remain
exactly zero instead of carrying unused initialization noise.

Reusable unconditional contexts follow the archived implementation's separate
precomputation protocol: a single-space prompt with offsets enabled for the
first 20 interior CLIP positions. The official paper does not specify this
cache optimization. The explicit mask is necessary because a blank prompt has
no semantic tokens; applying the conditional mask rule would silently produce
zero trainable offsets. Context checkpoints record both the prompt and
trainable-position count in their metadata.

For each variant, optimization stops once its semantic cosine similarity is
within `0.01` of `kappa`. Converged offset rows are frozen while remaining rows
continue. Both SD3.5 CLIP encoders use the same objective independently, as
specified in the paper.

Variants belonging to one prompt reuse the same initial diffusion latent,
matching the archived main-experiment protocol. Different prompts in a batch
receive independently sampled latents. The embedding scheduler uses the
actual number of timesteps produced by the diffusion scheduler; this matters
for schedulers such as PNDM, which produce 51 timesteps when 50 inference steps
are requested.

## Source Of Defaults

The official eight-page IJCNN 2026 paper and the archived scripts are both
recorded. The paper describes direct token offsets, Adam, `Normal(0, 1e-4)`
initialization, semantic tolerance `1e-2`, at most 50 optimization steps,
`kappa=0.8`, and coarse-to-fine ratio `r=0.4`. Its SHA-256 is
`196dfb5aaac629896b225a770b72ce9b74158b425ad00edc8bed08cd00841a2b`.
Other local TPSO PDFs are not release references.

The official paper does not list a model-specific diversity weight. For
release reproduction, archived Table I values and run records identify
`lambda=1` for SD1.5 and SD2.1 and `lambda=10` for SD3.5. Paper Table V is a
separate SD1.5 ablation over `lambda=0`, `5`, and `10`.

For result compatibility, executable behavior takes precedence where it differs:
RMSprop, token-count-scaled offsets, token masking, cached active
sample optimization, and 200 maximum optimization steps for SD3.5. The public
default remains `kappa=0.8`, matching the archived main-result directory names.

The scheduler uses the archived endpoint-inclusive `linspace` over the first
`int(T*r)` inference steps. For example, `T=10` and `r=0.4` gives optimized
weights `[1, 2/3, 1/3, 0, 0, ...]`.

SD3.5 uses 512-pixel output, T5 sequence length 77, 35 denoising steps, and the
archived stochastic-flow scheduler with `noise_scale=0.03`.

SD2.1 is generated at 768 pixels and resized to 512 pixels by the paper benchmark;
SD3.5 is generated directly at 512 pixels.

The paper does not state the number of denoising steps. The archived executable
path uses 35 steps for SD3.5 and 50 for SD1.5/SD2.1. The 28-step branch belongs
to SD3, not SD3.5.
