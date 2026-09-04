import pytest

from tpso.config import MODEL_SPECS, TPSOConfig, default_config, load_config


def test_release_defaults_are_model_specific():
    assert default_config("sd15").diversity_weight == 1.0
    assert default_config("sd21").diversity_weight == 1.0
    assert default_config("sd35").diversity_weight == 10.0
    assert {spec.name for spec in MODEL_SPECS.values()} == {"sd15", "sd21", "sd35"}
    assert all(len(spec.model_revision) == 40 for spec in MODEL_SPECS.values())
    assert MODEL_SPECS["sd21"].model_id == "sd2-community/stable-diffusion-2-1"
    assert MODEL_SPECS["sd35"].inference_steps == 35
    assert MODEL_SPECS["sd35"].image_size == 512


def test_camera_ready_optimization_defaults():
    for model in MODEL_SPECS:
        config = default_config(model)
        assert config.kappa == 0.8
        assert config.tolerance == 0.01
        assert config.scheduler_ratio == 0.4
        assert config.learning_rate == 0.01
        assert config.max_steps == (200 if model == "sd35" else 50)
        assert config.optimizer == "rmsprop"
        assert config.offset_init == "approx_zero"
        assert config.offset_std == 1e-4


def test_invalid_config_is_rejected():
    try:
        TPSOConfig(kappa=0.0).validate()
    except ValueError as error:
        assert "kappa" in str(error)
    else:
        raise AssertionError("Invalid kappa was accepted.")


def test_non_finite_config_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        TPSOConfig(diversity_weight=float("nan")).validate()


def test_fine_to_coarse_ratio_and_table_four_initializers_are_valid():
    assert TPSOConfig(scheduler_ratio=-0.6).validate().scheduler_ratio == -0.6
    assert TPSOConfig(offset_init="gamma_4_sqrt4").validate().offset_init == "gamma_4_sqrt4"


def test_yaml_config_is_loaded_without_mutating_input(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  name: sd15\n  inference_steps: 20\n"
        "tpso:\n  optimizer: adam\n  kappa: 0.9\n",
        encoding="utf-8",
    )
    first_spec, first_config = load_config(path, "sd15")
    second_spec, second_config = load_config(path, "sd15")
    assert first_spec.inference_steps == second_spec.inference_steps == 20
    assert first_config.kappa == second_config.kappa == 0.9


def test_yaml_config_rejects_context_path_traversal(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  name: sd15\n  context_filename: ../outside.pt\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local .pt filename"):
        load_config(path, "sd15")
