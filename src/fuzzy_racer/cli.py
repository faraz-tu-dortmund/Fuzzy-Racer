"""Command-line interface: ``python -m fuzzy_racer <command> [options]``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .levels import AI_MODE, PLAYER_MODE, TRAINING_MODE
from .network import Network, load_pretrained_network

MODES = {"ai": AI_MODE, "player": PLAYER_MODE}
TRAIN_MODES = {"ai": TRAINING_MODE, "player": PLAYER_MODE}


def _network(path: Path | None) -> Network:
    return Network.load(path) if path else load_pretrained_network()


MODE_HELP = "course to drive: the endless training course or the 5-level campaign"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for all sub-commands."""
    parser = argparse.ArgumentParser(
        prog="fuzzy_racer",
        description="Dodge traffic yourself or watch a NEAT-trained network do it.",
    )
    commands = parser.add_subparsers(dest="command", metavar="command")

    commands.add_parser("menu", help="open the graphical launcher (default)")

    play = commands.add_parser("play", help="drive yourself with the arrow keys")
    play.add_argument("--seed", type=int, help="seed for obstacle placement")

    ai = commands.add_parser("ai", help="watch a trained network drive")
    ai.add_argument("--network", type=Path, help="network JSON (default: pretrained)")
    ai.add_argument("--seed", type=int, help="seed for obstacle placement")
    ai.add_argument("--mode", choices=MODES, default="ai", help=MODE_HELP)

    train = commands.add_parser("train", help="evolve a new network with NEAT")
    train.add_argument("--generations", type=int, default=1000)
    train.add_argument("--config", type=Path, help="NEAT config file")
    train.add_argument("--mode", choices=TRAIN_MODES, default="ai", help=MODE_HELP)
    train.add_argument("--output-dir", type=Path, default=Path("results/training"))
    train.add_argument("--headless", action="store_true", help="train without a window")
    train.add_argument("--seed", type=int, help="seed for reproducible training")
    train.add_argument("--max-frames", type=int, help="frame cap per generation")

    draw = commands.add_parser("draw-network", help="save a diagram of a network")
    draw.add_argument("--network", type=Path, help="network JSON (default: pretrained)")
    draw.add_argument(
        "--output", type=Path, default=Path("results/pretrained_network.png")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command given on the command line and return an exit code."""
    args = build_parser().parse_args(argv)

    if args.command in (None, "menu"):
        from .menu import run_menu

        run_menu()
    elif args.command == "play":
        from .game import play

        print(play(seed=args.seed))
    elif args.command == "ai":
        from .game import watch_ai

        print(watch_ai(_network(args.network), mode=MODES[args.mode], seed=args.seed))
    elif args.command == "train":
        from .training import train

        result = train(
            args.generations,
            config_path=args.config,
            mode=TRAIN_MODES[args.mode],
            output_dir=args.output_dir,
            headless=args.headless,
            seed=args.seed,
            max_frames=args.max_frames,
        )
        print(result)
    elif args.command == "draw-network":
        from .visualize import plot_network

        print(f"Diagram saved to {plot_network(_network(args.network), args.output)}")
    return 0
