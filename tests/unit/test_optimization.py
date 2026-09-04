import pytest
import torch

from tpso.config import TPSOConfig
from tpso.optimization import EncodedPrompt, initialize_offsets, optimize_prompt_offsets


def test_offset_optimization_reaches_semantic_band():
    batch = 4
    reference = torch.tensor([[1.0, 0.0]]).repeat(batch, 1)

    def encode(offsets):
        projected = reference + offsets
        return EncodedPrompt(hidden=projected.unsqueeze(1), projected=projected)

    result = optimize_prompt_offsets(
        encode,
        EncodedPrompt(hidden=reference.unsqueeze(1), projected=reference),
        (batch, 2),
        num_variants=2,
        config=TPSOConfig(
            kappa=0.8,
            tolerance=0.02,
            diversity_weight=0.0,
            learning_rate=0.1,
            max_steps=100,
            offset_std=1e-2,
        ),
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )
    assert result.converged.all()
    assert torch.all((result.cosine_similarity - 0.8).abs() < 0.02)


def test_offset_optimization_rejects_rows_without_trainable_tokens():
    reference = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    def encode(offsets):
        projected = reference + offsets.sum(dim=1)
        return EncodedPrompt(hidden=projected.unsqueeze(1), projected=projected)

    mask = torch.tensor([[[1.0]], [[0.0]]])
    with pytest.raises(ValueError, match="blank conditional prompts"):
        optimize_prompt_offsets(
            encode,
            EncodedPrompt(hidden=reference.unsqueeze(1), projected=reference),
            (2, 1, 2),
            num_variants=1,
            config=TPSOConfig(),
            device="cpu",
            trainable_mask=mask,
        )


def test_offset_optimization_zeros_frozen_offsets():
    reference = torch.tensor([[1.0, 0.0]])

    def encode(offsets):
        projected = reference + offsets.sum(dim=1)
        return EncodedPrompt(hidden=projected.unsqueeze(1), projected=projected)

    result = optimize_prompt_offsets(
        encode,
        EncodedPrompt(hidden=reference.unsqueeze(1), projected=reference),
        (1, 3, 2),
        num_variants=1,
        config=TPSOConfig(max_steps=1),
        device="cpu",
        initial_offsets=torch.ones(1, 3, 2),
        trainable_mask=torch.tensor([[[1.0], [0.0], [0.0]]]),
    )

    torch.testing.assert_close(result.offsets[:, 1:], torch.zeros(1, 2, 2))


def test_offset_optimization_only_reencodes_active_rows():
    reference = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    subset_sizes = []

    def encode(offsets):
        projected = reference + offsets
        return EncodedPrompt(hidden=projected.unsqueeze(1), projected=projected)

    def encode_subset(offsets, indices):
        subset_sizes.append(indices.numel())
        projected = reference.index_select(0, indices) + offsets
        already_converged = torch.tensor([0.8, 0.6]).expand_as(projected)
        projected = torch.where(
            (indices == 0).unsqueeze(1), already_converged, projected
        )
        return EncodedPrompt(hidden=projected.unsqueeze(1), projected=projected)

    optimize_prompt_offsets(
        encode,
        EncodedPrompt(hidden=reference.unsqueeze(1), projected=reference),
        (2, 2),
        num_variants=1,
        config=TPSOConfig(
            kappa=0.8,
            tolerance=0.02,
            diversity_weight=0.0,
            learning_rate=0.1,
            max_steps=100,
            offset_std=1e-2,
        ),
        device="cpu",
        generator=torch.Generator().manual_seed(7),
        encode_subset=encode_subset,
    )

    assert subset_sizes[0] == 2
    assert min(subset_sizes) == 1


def test_one_step_result_keeps_source_pre_update_representation():
    reference = torch.tensor([[1.0, 0.0]])
    initial = torch.tensor([[0.0, 1.0]])

    def encode(offsets):
        projected = reference + offsets
        return EncodedPrompt(hidden=projected.unsqueeze(1), projected=projected)

    result = optimize_prompt_offsets(
        encode,
        EncodedPrompt(hidden=reference.unsqueeze(1), projected=reference),
        (1, 2),
        num_variants=1,
        config=TPSOConfig(
            kappa=0.8,
            diversity_weight=0.0,
            learning_rate=0.1,
            max_steps=1,
        ),
        device="cpu",
        initial_offsets=initial,
    )

    torch.testing.assert_close(result.optimized.projected, reference + initial)
    torch.testing.assert_close(result.offsets, initial)


def test_full_batch_mode_reencodes_every_row_each_step():
    reference = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    batch_sizes = []

    def encode(offsets):
        batch_sizes.append(offsets.shape[0])
        projected = reference + offsets
        return EncodedPrompt(hidden=projected.unsqueeze(1), projected=projected)

    result = optimize_prompt_offsets(
        encode,
        EncodedPrompt(hidden=reference.unsqueeze(1), projected=reference),
        (2, 2),
        num_variants=2,
        config=TPSOConfig(
            kappa=0.5,
            diversity_weight=1.0,
            learning_rate=0.01,
            max_steps=3,
        ),
        device="cpu",
        initial_offsets=torch.tensor([[0.0, 0.2], [0.0, -0.2]]),
        active_subset=False,
    )

    assert result.steps == 3
    assert batch_sizes == [2, 2, 2]


@pytest.mark.parametrize(
    "offset_init,offset_std",
    [
        ("approx_zero", 1e-4),
        ("normal", 1e-4),
        ("normal", 1.0),
        ("zero", 1e-4),
        ("normal_1", 1e-4),
        ("normal_2", 1e-4),
        ("laplace_1", 1e-4),
        ("laplace_1.5", 1e-4),
        ("laplace_sqrt1.5", 1e-4),
        ("gamma_1_1", 1e-4),
        ("gamma_4_4", 1e-4),
        ("gamma_4_sqrt4", 1e-4),
        ("gamma_7_sqrt7", 1e-4),
        ("uniform_0_1", 1e-4),
        ("uniform_-1_1", 1e-4),
        ("uniform_-3_3", 1e-4),
    ],
)
def test_table_four_initializers_are_finite_and_reproducible(offset_init, offset_std):
    config = TPSOConfig(offset_init=offset_init, offset_std=offset_std)
    first = initialize_offsets(
        (64, 8),
        config,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(11),
    )
    second = initialize_offsets(
        (64, 8),
        config,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(11),
    )

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    if offset_init == "uniform_0_1":
        assert first.min() >= 0 and first.max() <= 1
    elif offset_init == "uniform_-1_1":
        assert first.min() >= -1 and first.max() <= 1
    elif offset_init == "uniform_-3_3":
        assert first.min() >= -3 and first.max() <= 3
