"""A small, dependency-free feed-forward network that can drive the car."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .assets import PRETRAINED_NETWORK_PATH
from .sprites import Steer

#: Names of the network inputs, in the order produced by ``Race.sensors``.
INPUT_NAMES = ("car x", "gap to right edge", "obstacle x", "obstacle y", "level")
#: Names of the network outputs.
OUTPUT_NAMES = ("steer",)


def _sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


def _tanh(z: float) -> float:
    return math.tanh(max(-60.0, min(60.0, 2.5 * z)))


def _relu(z: float) -> float:
    return z if z > 0.0 else 0.0


def _identity(z: float) -> float:
    return z


def _clamped(z: float) -> float:
    return max(-1.0, min(1.0, z))


ACTIVATIONS: dict[str, Callable[[float], float]] = {
    "sigmoid": _sigmoid,
    "tanh": _tanh,
    "relu": _relu,
    "identity": _identity,
    "clamped": _clamped,
}

AGGREGATIONS: dict[str, Callable[[Sequence[float]], float]] = {
    "sum": sum,
    "product": lambda values: math.prod(values, start=1.0),
    "max": max,
    "min": min,
    "mean": statistics.fmean,
}


@dataclass(frozen=True)
class Node:
    """One evaluated node: ``activation(bias + response * aggregation(inputs))``.

    Attributes:
        key: The node id; input nodes have negative ids.
        activation: Name of the activation function.
        aggregation: Name of the function combining weighted inputs.
        bias: Bias added before activation.
        response: Multiplier applied to the aggregated input.
        links: ``(source node id, weight)`` pairs feeding this node.
    """

    key: int
    activation: str
    aggregation: str
    bias: float
    response: float
    links: tuple[tuple[int, float], ...]


class Network:
    """A feed-forward network whose nodes are evaluated in topological order.

    Args:
        input_keys: Ids of the input nodes.
        output_keys: Ids of the output nodes.
        nodes: Hidden and output nodes, sorted so every node comes after all
            nodes it depends on.
    """

    def __init__(
        self,
        input_keys: Sequence[int],
        output_keys: Sequence[int],
        nodes: Sequence[Node],
    ) -> None:
        unknown = {n.activation for n in nodes} - ACTIVATIONS.keys()
        unknown |= {n.aggregation for n in nodes} - AGGREGATIONS.keys()
        if unknown:
            raise ValueError(f"Unsupported functions: {sorted(unknown)}")
        self.input_keys = tuple(input_keys)
        self.output_keys = tuple(output_keys)
        self.nodes = tuple(nodes)

    def activate(self, inputs: Sequence[float]) -> list[float]:
        """Feed ``inputs`` through the network and return the output values."""
        if len(inputs) != len(self.input_keys):
            raise ValueError(
                f"Expected {len(self.input_keys)} inputs, got {len(inputs)}"
            )
        values: dict[int, float] = dict.fromkeys(self.output_keys, 0.0)
        values.update(zip(self.input_keys, inputs, strict=True))
        for node in self.nodes:
            weighted = [values[source] * weight for source, weight in node.links]
            total = AGGREGATIONS[node.aggregation](weighted)
            values[node.key] = ACTIVATIONS[node.activation](
                node.bias + node.response * total
            )
        return [values[key] for key in self.output_keys]

    def decide(self, inputs: Sequence[float]) -> Steer:
        """Turn the first output into a steering command.

        Outputs above 0.5 steer left, below 0.5 steer right, and exactly 0.5
        keeps the car straight.
        """
        return steer_from_output(self.activate(inputs)[0])

    # --- conversion -------------------------------------------------------

    @classmethod
    def from_neat(cls, net: Any) -> Network:
        """Convert a ``neat.nn.FeedForwardNetwork`` into a :class:`Network`."""
        nodes = [
            Node(
                key=key,
                activation=_function_name(activation, "_activation"),
                aggregation=_function_name(aggregation, "_aggregation"),
                bias=bias,
                response=response,
                links=tuple((source, weight) for source, weight in links),
            )
            for key, activation, aggregation, bias, response, links in net.node_evals
        ]
        return cls(net.input_nodes, net.output_nodes, nodes)

    @classmethod
    def from_genome(cls, genome: Any, config: Any) -> Network:
        """Build the network encoded by a NEAT genome."""
        import neat

        return cls.from_neat(neat.nn.FeedForwardNetwork.create(genome, config))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the network."""
        return {
            "inputs": list(self.input_keys),
            "outputs": list(self.output_keys),
            "nodes": [asdict(node) for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Network:
        """Rebuild a network from :meth:`to_dict` output."""
        nodes = [
            Node(**{**node, "links": tuple(tuple(link) for link in node["links"])})
            for node in data["nodes"]
        ]
        return cls(data["inputs"], data["outputs"], nodes)

    def save(self, path: str | Path) -> Path:
        """Write the network to a JSON file and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Network:
        """Read a network from a JSON file written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def steer_from_output(output: float) -> Steer:
    """Map a network output in ``[0, 1]`` to a steering command."""
    if output > 0.5:
        return Steer.LEFT
    if output < 0.5:
        return Steer.RIGHT
    return Steer.STRAIGHT


def _function_name(function: Callable[..., Any], suffix: str) -> str:
    name = function.__name__
    return name[: -len(suffix)] if name.endswith(suffix) else name


def load_pretrained_network() -> Network:
    """Load the network trained for the original project."""
    return Network.load(PRETRAINED_NETWORK_PATH)
