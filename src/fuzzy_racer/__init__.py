"""Fuzzy Racer: a traffic-dodging game with a NEAT-trained autopilot."""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from .game import GameResult, play, run_game, watch_ai  # noqa: E402
from .levels import (  # noqa: E402
    AI_MODE,
    PLAYER_MODE,
    TRAINING_MODE,
    GameMode,
    LevelSpec,
    ObstacleCourse,
    Pattern,
)
from .network import Network, load_pretrained_network  # noqa: E402
from .race import FrameResult, Race  # noqa: E402
from .sprites import Car, Obstacle, Road, Steer  # noqa: E402
from .training import TrainingResult, load_config, train  # noqa: E402
from .visualize import plot_network  # noqa: E402

__version__ = "1.0.0"

__all__ = [
    "AI_MODE",
    "PLAYER_MODE",
    "TRAINING_MODE",
    "Car",
    "FrameResult",
    "GameMode",
    "GameResult",
    "LevelSpec",
    "Network",
    "Obstacle",
    "ObstacleCourse",
    "Pattern",
    "Race",
    "Road",
    "Steer",
    "TrainingResult",
    "load_config",
    "load_pretrained_network",
    "play",
    "plot_network",
    "run_game",
    "train",
    "watch_ai",
]
