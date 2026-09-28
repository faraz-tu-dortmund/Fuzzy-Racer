import pytest

from fuzzy_racer import Network, Steer, load_pretrained_network
from fuzzy_racer.network import Node, steer_from_output


def test_pretrained_network_matches_original_outputs():
    # Reference values computed with neat-python 0.92 from the original pickle.
    network = load_pretrained_network()
    assert network.activate((265, 335, 100, 0, 1))[0] == pytest.approx(8.7565e-27)
    assert network.activate((265, 335, 300, 400, 2))[0] == pytest.approx(1.0)


def test_json_round_trip(tmp_path):
    network = load_pretrained_network()
    path = network.save(tmp_path / "net.json")
    loaded = Network.load(path)
    inputs = (120.0, 480.0, 300.0, 250.0, 2)
    assert loaded.activate(inputs) == network.activate(inputs)


def test_single_node_network():
    node = Node(0, "identity", "sum", 1.0, 2.0, ((-1, 0.5), (-2, -1.0)))
    network = Network([-1, -2], [0], [node])
    assert network.activate([4.0, 1.0]) == [1.0 + 2.0 * (2.0 - 1.0)]


def test_wrong_input_count_raises():
    with pytest.raises(ValueError):
        load_pretrained_network().activate([1.0, 2.0])


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        Network([-1], [0], [Node(0, "nope", "sum", 0.0, 1.0, ((-1, 1.0),))])


@pytest.mark.parametrize(
    ("output", "expected"),
    [(0.9, Steer.LEFT), (0.1, Steer.RIGHT), (0.5, Steer.STRAIGHT)],
)
def test_steer_from_output(output, expected):
    assert steer_from_output(output) is expected
