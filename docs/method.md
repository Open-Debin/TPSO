# Method

TPSO adds trainable offsets to CLIP token embeddings. It optimizes semantic
retention and pairwise diversity without updating diffusion-model weights.

| Component | Implementation |
| --- | --- |
| Token offsets | `tpso.text_encoder.CLIPOffsetSession` |
| Semantic and diversity losses | `tpso.losses` |
| Offset optimization | `tpso.optimization.optimize_prompt_offsets` |
| Coarse-to-fine scheduler | `tpso.scheduling` |
| SD1.5/2.1 inference | `tpso.pipelines.stable_diffusion` |
| SD3.5 inference | `tpso.pipelines.stable_diffusion3` |

Default settings are `kappa=0.8`, tolerance `0.01`, learning rate `0.01`, and
coarse-to-fine ratio `0.4`. The release follows the archived executable
implementation: RMSprop, token-count-scaled offsets, active-sample optimization,
and 35 denoising steps for SD3.5.

Variants of one prompt share an initial diffusion latent. Different prompts use
independent latents. SD3.5 optimizes both CLIP encoders and leaves T5 unchanged.
