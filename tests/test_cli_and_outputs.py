import pytest

from fuzzy_racer import plot_network, train
from fuzzy_racer.cli import build_parser, main
from fuzzy_racer.network import load_pretrained_network


def test_parser_defaults():
    args = build_parser().parse_args(["train", "--headless", "--generations", "3"])
    assert args.headless and args.generations == 3
    assert build_parser().parse_args([]).command is None


def test_network_plot_is_written(tmp_path):
    assert plot_network(load_pretrained_network(), tmp_path / "n.png").exists()


def test_draw_network_command(tmp_path):
    assert main(["draw-network", "--output", str(tmp_path / "net.png")]) == 0
    assert (tmp_path / "net.png").exists()


@pytest.fixture
def small_config(tmp_path):
    from fuzzy_racer.assets import NEAT_CONFIG_PATH

    text = NEAT_CONFIG_PATH.read_text().replace(
        "pop_size              = 500", "pop_size              = 12"
    )
    path = tmp_path / "config.ini"
    path.write_text(text)
    return path


def test_headless_training_smoke(tmp_path, small_config):
    result = train(
        2,
        config_path=small_config,
        output_dir=tmp_path,
        headless=True,
        seed=0,
        max_frames=400,
        verbose=False,
    )
    assert result.generations == 2
    for path in result.files.values():
        assert path.exists()
