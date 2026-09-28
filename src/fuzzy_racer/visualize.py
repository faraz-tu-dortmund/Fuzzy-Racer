"""Diagrams of trained networks.

All functions save a PNG file and return its path; nothing is shown on
screen, so they work on machines without a display.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from .network import INPUT_NAMES, OUTPUT_NAMES, Network

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
BLUE = "#2a78d6"
RED = "#e34948"


def _new_figure(width: float = 8, height: float = 4.5) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE)
    axes = figure.add_subplot()
    axes.set_facecolor(SURFACE)
    return figure, axes


def _save(figure: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    return path


def _layers(network: Network) -> dict[int, int]:
    """Assign each node a column: inputs 0, outputs last, hidden in between."""
    depth = dict.fromkeys(network.input_keys, 0)
    for node in network.nodes:
        depth[node.key] = 1 + max((depth.get(s, 0) for s, _ in node.links), default=0)
    last = max([depth[k] for k in network.output_keys] + [1])
    for key in network.output_keys:
        depth[key] = last
    return depth


def plot_network(
    network: Network,
    path: str | Path,
    input_names: Sequence[str] = INPUT_NAMES,
    output_names: Sequence[str] = OUTPUT_NAMES,
) -> Path:
    """Draw the network's topology: nodes by layer, links by sign and weight."""
    depth = _layers(network)
    columns: dict[int, list[int]] = {}
    keys = [*network.input_keys, *(n.key for n in network.nodes)]
    keys += [key for key in network.output_keys if key not in keys]
    for key in keys:
        columns.setdefault(depth[key], []).append(key)
    position = {}
    for column, keys in columns.items():
        for row, key in enumerate(keys):
            position[key] = (column, (len(keys) - 1) / 2 - row)
    names = dict(zip(network.input_keys, input_names, strict=False))
    names.update(zip(network.output_keys, output_names, strict=False))

    figure, axes = _new_figure(8, 5)
    largest = max((abs(w) for n in network.nodes for _, w in n.links), default=1.0)
    for node in network.nodes:
        for source, weight in node.links:
            (x0, y0), (x1, y1) = position[source], position[node.key]
            axes.plot(
                [x0, x1],
                [y0, y1],
                color=BLUE if weight > 0 else RED,
                lw=0.8 + 3.2 * abs(weight) / largest,
                alpha=0.9,
                zorder=1,
                solid_capstyle="round",
            )
    for key, (x, y) in position.items():
        if key in network.input_keys:
            axes.add_patch(
                FancyBboxPatch(
                    (x - 0.09, y - 0.13),
                    0.18,
                    0.26,
                    boxstyle="round,pad=0,rounding_size=0.04",
                    fc=NEUTRAL,
                    ec=AXIS,
                    zorder=2,
                )
            )
            axes.annotate(
                names[key],
                (x - 0.14, y),
                ha="right",
                va="center",
                color=TEXT_SECONDARY,
                fontsize=9,
            )
        else:
            is_output = key in network.output_keys
            axes.scatter(
                x,
                y,
                s=420,
                zorder=2,
                color=BLUE if is_output else SURFACE,
                edgecolors=BLUE,
                linewidths=1.5,
            )
            if is_output:
                axes.annotate(
                    names[key],
                    (x + 0.14, y),
                    ha="left",
                    va="center",
                    color=TEXT_SECONDARY,
                    fontsize=9,
                )
    axes.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2.5, label="positive weight"),
            Line2D([], [], color=RED, lw=2.5, label="negative weight"),
        ],
        frameon=False,
        labelcolor=TEXT_SECONDARY,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    axes.set_xlim(-1.1, max(columns) + 0.9)
    axes.set_ylim(
        -max(len(k) for k in columns.values()) / 2 - 0.3,
        max(len(k) for k in columns.values()) / 2 + 0.1,
    )
    axes.set_axis_off()
    axes.set_title(
        "Network topology (line width = |weight|)", loc="left", color=TEXT, fontsize=12
    )
    return _save(figure, path)
