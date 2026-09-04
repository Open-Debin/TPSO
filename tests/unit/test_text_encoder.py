from types import SimpleNamespace

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel

from tpso.text_encoder import prepare_clip_session


class TinyTokenizer:
    model_max_length = 6

    def __call__(self, prompts, **_kwargs):
        rows = [[98, 5, 6, 99, 0, 0] for _ in prompts]
        return SimpleNamespace(input_ids=torch.tensor(rows))


def test_clip_offsets_are_differentiable_and_records_gradient_mask():
    config = CLIPTextConfig(
        vocab_size=100,
        hidden_size=8,
        intermediate_size=16,
        projection_dim=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=6,
        bos_token_id=98,
        eos_token_id=99,
        pad_token_id=0,
    )
    text_model = CLIPTextModel(config).text_model
    projection = torch.nn.Linear(8, 8, bias=False)
    session = prepare_clip_session(
        TinyTokenizer(), text_model, projection, ["ignored"], 2, device="cpu"
    )
    offsets = torch.randn(session.offset_shape, requires_grad=True)
    encoded = session.encode(offsets)
    encoded.projected.sum().backward()

    assert encoded.hidden.shape == (2, 6, 8)
    assert offsets.grad is not None
    assert offsets.grad[:, 2:].abs().sum() > 0
    assert session.token_mask[:, :2].all()
    assert not session.token_mask[:, 2:].any()


def test_explicit_token_count_overrides_blank_prompt_mask():
    config = CLIPTextConfig(
        vocab_size=100,
        hidden_size=8,
        intermediate_size=16,
        projection_dim=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=6,
        bos_token_id=98,
        eos_token_id=99,
        pad_token_id=0,
    )
    text_model = CLIPTextModel(config).text_model
    session = prepare_clip_session(
        TinyTokenizer(),
        text_model,
        torch.nn.Identity(),
        " ",
        3,
        device="cpu",
        trainable_token_count=3,
    )

    assert session.token_mask.shape == (3, 4, 1)
    assert session.token_mask[:, :3].all()
    assert not session.token_mask[:, 3:].any()
    assert torch.allclose(session.offset_scale, torch.full((3, 1, 1), 1 / 3))


def test_whitespace_token_mode_matches_sd12_source_scaling():
    config = CLIPTextConfig(
        vocab_size=100,
        hidden_size=8,
        intermediate_size=16,
        projection_dim=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=6,
        bos_token_id=98,
        eos_token_id=99,
        pad_token_id=0,
    )
    session = prepare_clip_session(
        TinyTokenizer(),
        CLIPTextModel(config).text_model,
        torch.nn.Identity(),
        "two words",
        2,
        device="cpu",
        token_count_mode="whitespace",
    )
    assert session.token_mask[:, :2].all()
    assert not session.token_mask[:, 2:].any()
    assert torch.allclose(session.offset_scale, torch.full((2, 1, 1), 0.5))


@pytest.mark.parametrize("hidden_state_index", [None, -2])
def test_zero_offsets_match_transformers_clip_forward(hidden_state_index):
    config = CLIPTextConfig(
        vocab_size=100,
        hidden_size=8,
        intermediate_size=16,
        projection_dim=5,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=6,
        bos_token_id=98,
        eos_token_id=99,
        pad_token_id=0,
    )
    text_model = CLIPTextModel(config).text_model.eval()
    projection = torch.nn.Linear(8, 5, bias=False).eval()
    session = prepare_clip_session(
        TinyTokenizer(),
        text_model,
        projection,
        ["ignored"],
        2,
        device="cpu",
        hidden_state_index=hidden_state_index,
    )

    with torch.no_grad():
        expected = text_model(
            input_ids=session.input_ids,
            output_hidden_states=hidden_state_index is not None,
            return_dict=True,
        )
        expected_hidden = (
            expected.hidden_states[hidden_state_index]
            if hidden_state_index is not None
            else expected.last_hidden_state
        )
        expected_projected = projection(expected.pooler_output)
        actual = session.original()

    torch.testing.assert_close(actual.hidden, expected_hidden)
    torch.testing.assert_close(actual.projected, expected_projected)
