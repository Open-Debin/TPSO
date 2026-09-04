import torch

from tpso.losses import diversity_loss, semantic_loss


def test_semantic_loss_matches_equation_five():
    cosine = torch.tensor([0.7, 0.8, 0.9])
    assert torch.isclose(semantic_loss(cosine, 0.8, 0.01), torch.tensor(0.18))


def test_diversity_loss_matches_source_average_pairwise_distance():
    projected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.isclose(diversity_loss(projected, 2), torch.tensor(-1.0))


def test_diversity_loss_compensates_for_inactive_rows():
    projected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    active = torch.tensor([True, False])
    assert torch.isclose(diversity_loss(projected, 2, active), torch.tensor(-2.0))


def test_diversity_loss_rejects_incomplete_groups():
    try:
        diversity_loss(torch.ones(3, 2), 2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Incomplete variant group was accepted.")
