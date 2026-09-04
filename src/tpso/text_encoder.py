"""CLIP token-offset encoding used by all supported TPSO backbones."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .optimization import EncodedPrompt


def _causal_attention_mask(
    batch_size: int, sequence_length: int, tensor: torch.Tensor
) -> torch.Tensor:
    mask = torch.full(
        (batch_size, 1, sequence_length, sequence_length),
        float("-inf"),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.triu(mask, diagonal=1)


@dataclass
class CLIPOffsetSession:
    """A frozen CLIP encoder with a batch of trainable token offsets."""

    text_model: torch.nn.Module
    projection: torch.nn.Module
    input_ids: torch.Tensor
    token_embeddings: torch.Tensor
    token_mask: torch.Tensor
    offset_scale: torch.Tensor
    hidden_state_index: int | None = None
    original_encoded: EncodedPrompt | None = None

    @property
    def offset_shape(self) -> tuple[int, int, int]:
        batch, length, width = self.token_embeddings.shape
        return batch, length - 2, width

    def _encode(
        self, offsets: torch.Tensor, indices: torch.Tensor | None = None
    ) -> EncodedPrompt:
        expected_shape = self.offset_shape
        if indices is not None:
            if indices.ndim != 1 or indices.dtype != torch.long:
                raise TypeError("indices must be a one-dimensional torch.long tensor.")
            expected_shape = (indices.numel(), *expected_shape[1:])
        if tuple(offsets.shape) != expected_shape:
            raise ValueError(
                f"Expected offsets with shape {expected_shape}, got {tuple(offsets.shape)}."
            )
        input_ids = self.input_ids
        token_embeddings = self.token_embeddings
        offset_scale = self.offset_scale
        if indices is not None:
            input_ids = input_ids.index_select(0, indices)
            token_embeddings = token_embeddings.index_select(0, indices)
            offset_scale = offset_scale.index_select(0, indices)
        padded = torch.nn.functional.pad(offsets * offset_scale, (0, 0, 1, 1))
        inputs = token_embeddings + padded
        batch, length = input_ids.shape
        attention_mask = _causal_attention_mask(batch, length, inputs)
        # Transformers 5 folded the old ``causal_attention_mask`` argument into
        # ``attention_mask``. Calling layers directly preserves the archived
        # Transformers 4 computation on both APIs.
        hidden_states = [inputs] if self.hidden_state_index is not None else None
        encoded = inputs
        for layer in self.text_model.encoder.layers:
            encoded = layer(encoded, attention_mask=attention_mask, is_causal=True)
            if hidden_states is not None:
                hidden_states.append(encoded)
        final_hidden = self.text_model.final_layer_norm(encoded)
        input_ids = input_ids.to(dtype=torch.int)
        if self.text_model.eos_token_id == 2:
            eos_positions = input_ids.argmax(dim=-1)
        else:
            eos_positions = (input_ids == self.text_model.eos_token_id).int().argmax(dim=-1)
        pooled = final_hidden[torch.arange(batch, device=inputs.device), eos_positions]
        projected = self.projection(pooled)
        hidden = (
            hidden_states[self.hidden_state_index]
            if hidden_states is not None
            else final_hidden
        )
        return EncodedPrompt(hidden=hidden, projected=projected)

    def encode(self, offsets: torch.Tensor) -> EncodedPrompt:
        return self._encode(offsets)

    def encode_subset(
        self, offsets: torch.Tensor, indices: torch.Tensor
    ) -> EncodedPrompt:
        """Encode only unconverged rows during prompt-offset optimization."""

        return self._encode(offsets, indices)

    def original(self) -> EncodedPrompt:
        if self.original_encoded is not None:
            return self.original_encoded
        offsets = torch.zeros(
            self.offset_shape,
            device=self.token_embeddings.device,
            dtype=self.token_embeddings.dtype,
        )
        with torch.no_grad():
            return self.encode(offsets)


def prepare_clip_session(
    tokenizer: object,
    text_model: torch.nn.Module,
    projection: torch.nn.Module,
    prompts: str | Sequence[str],
    num_variants: int,
    *,
    device: torch.device | str,
    hidden_state_index: int | None = None,
    trainable_token_count: int | None = None,
    token_count_mode: str = "tokenizer",
) -> CLIPOffsetSession:
    """Tokenize prompts and prepare a frozen encoder session for TPSO."""

    if num_variants <= 0:
        raise ValueError("num_variants must be positive.")
    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompt_list:
        raise ValueError("At least one prompt is required.")
    repeated = [prompt for prompt in prompt_list for _ in range(num_variants)]
    max_length = min(int(getattr(tokenizer, "model_max_length", 77)), 77)
    tokenized = tokenizer(
        repeated,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)

    text_model = text_model.to(device=device, dtype=torch.float32).eval()
    projection = projection.to(device=device, dtype=torch.float32).eval()
    text_model.requires_grad_(False)
    projection.requires_grad_(False)
    with torch.no_grad():
        token_embeddings = text_model.embeddings(input_ids).detach()

    if token_count_mode not in {"tokenizer", "whitespace"}:
        raise ValueError("token_count_mode must be 'tokenizer' or 'whitespace'.")
    token_mask = torch.zeros(
        (len(repeated), max_length - 2, 1),
        device=device,
        dtype=token_embeddings.dtype,
    )
    if trainable_token_count is not None:
        if isinstance(trainable_token_count, bool) or not isinstance(
            trainable_token_count, int
        ):
            raise TypeError("trainable_token_count must be an integer or None.")
        if not 1 <= trainable_token_count <= max_length - 2:
            raise ValueError(
                f"trainable_token_count must be between 1 and {max_length - 2}."
            )
        counts = [trainable_token_count] * len(repeated)
    elif token_count_mode == "whitespace":
        counts = [min(len(prompt.split()), max_length - 2) for prompt in repeated]
    else:
        eos_positions = input_ids.to(dtype=torch.int).argmax(dim=-1)
        counts = [max(0, eos - 1) for eos in eos_positions.tolist()]
    if any(count <= 0 for count in counts):
        raise ValueError(
            "Every optimized prompt must contain at least one trainable token."
        )
    for row, count in enumerate(counts):
        token_mask[row, :count] = 1
    offset_scale = torch.tensor(
        [1.0 / count for count in counts],
        device=device,
        dtype=token_embeddings.dtype,
    ).reshape(-1, 1, 1)
    session = CLIPOffsetSession(
        text_model=text_model,
        projection=projection,
        input_ids=input_ids,
        token_embeddings=token_embeddings,
        token_mask=token_mask,
        offset_scale=offset_scale,
        hidden_state_index=hidden_state_index,
    )
    unique_indices = torch.arange(
        0, len(repeated), num_variants, device=input_ids.device
    )
    unique_session = CLIPOffsetSession(
        text_model=text_model,
        projection=projection,
        input_ids=input_ids.index_select(0, unique_indices),
        token_embeddings=token_embeddings.index_select(0, unique_indices),
        token_mask=token_mask.index_select(0, unique_indices),
        offset_scale=offset_scale.index_select(0, unique_indices),
        hidden_state_index=hidden_state_index,
    )
    with torch.no_grad():
        unique_original = unique_session.original()
    session.original_encoded = EncodedPrompt(
        hidden=unique_original.hidden.repeat_interleave(num_variants, dim=0),
        projected=unique_original.projected.repeat_interleave(num_variants, dim=0),
    )
    return session
