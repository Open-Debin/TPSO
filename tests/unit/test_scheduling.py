import torch

from tpso.scheduling import coarse_to_fine_alphas, schedule_embeddings


def test_paper_ratio_only_changes_early_steps():
    alphas = coarse_to_fine_alphas(10, 0.4)
    assert torch.allclose(alphas[:4], torch.tensor([1.0, 2 / 3, 1 / 3, 0.0]))
    assert torch.count_nonzero(alphas[4:]) == 0


def test_ratio_endpoints_follow_equation_ten():
    assert torch.count_nonzero(coarse_to_fine_alphas(10, 0.0)) == 0
    assert torch.allclose(
        coarse_to_fine_alphas(5, 1.0),
        torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
    )


def test_embedding_schedule_starts_optimized_and_ends_original():
    optimized = torch.ones(2, 3)
    original = torch.zeros(2, 3)
    scheduled = schedule_embeddings(optimized, original, 5, 0.4)
    assert torch.equal(scheduled[:, 0], optimized)
    assert torch.equal(scheduled[:, -1], original)


def test_negative_ratio_reverses_schedule_for_fine_to_coarse_ablation():
    positive = coarse_to_fine_alphas(10, 0.4)
    negative = coarse_to_fine_alphas(10, -0.4)

    assert torch.equal(negative, positive.flip(0))
    assert torch.count_nonzero(negative[:6]) == 0
    assert negative[-1] == 1
