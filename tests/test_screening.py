import pytest

from spaceship_titanic.screening import confirm, screen, seeded


def test_seeded_keeps_seed42_id_and_does_not_mutate():
    config = {"experiment_id": "x", "random_state": 42, "model_params": {"depth": 6}}
    assert seeded(config, 42)["experiment_id"] == "x"
    other = seeded(config, 123)
    assert other["experiment_id"] == "x_seed123" and other["random_state"] == 123
    other["model_params"]["depth"] = 8
    assert config["model_params"]["depth"] == 6 and config["random_state"] == 42


def test_screen_needs_mean_gain_and_two_of_three_positive():
    control = {42: 0.818, 123: 0.821, 2026: 0.816}
    assert screen(control, {42: 0.824, 123: 0.823, 2026: 0.815})["passes"]
    # Large mean driven by one seed fails the positive-seed requirement.
    assert not screen(control, {42: 0.830, 123: 0.820, 2026: 0.815})["passes"]
    # H-D-02 winner vs matched control: mean +0.00176 is below +0.002.
    control = {42: 0.818820, 123: 0.821351, 2026: 0.815714}
    result = screen(control, {42: 0.824111, 123: 0.817439, 2026: 0.819625})
    assert not result["passes"] and result["positive_seeds"] == 2


def test_confirm_requires_every_seed_positive_and_matching_seeds():
    assert confirm({7: 0.81, 99: 0.82}, {7: 0.812, 99: 0.821})["passes"]
    assert not confirm({7: 0.81, 99: 0.82}, {7: 0.812, 99: 0.819})["passes"]
    with pytest.raises(ValueError):
        confirm({7: 0.81}, {99: 0.82})
