"""Level definitions and the obstacle course that plays them out.

A level sends a fixed number of obstacles down the road in *waves*. The next
wave spawns once the previous one has travelled ``spawn_gap`` pixels, and the
level is cleared when every obstacle has left the screen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import pygame

from . import settings
from .assets import Sprites
from .sprites import Obstacle


class Pattern(Enum):
    """How the obstacles of a wave are chosen and placed."""

    #: One truck or van at a random position.
    RANDOM = "random"
    #: Trucks or vans snapped to five evenly spaced lanes.
    LANES = "lanes"
    #: A wide truck on the left, leaving a gap on the right.
    WIDE_TRUCK = "wide_truck"
    #: A truck and a bus with a narrow gap that alternates sides every wave.
    SLALOM = "slalom"


@dataclass(frozen=True)
class LevelSpec:
    """The rules of one level.

    Attributes:
        pattern: How obstacles are placed.
        speed: Obstacle speed in pixels per frame.
        spawn_gap: Distance the latest wave travels before the next spawns.
        obstacle_count: Total obstacles sent during the level.
        wave_size: Obstacles spawned per wave.
    """

    pattern: Pattern
    speed: int = settings.BASE_OBSTACLE_SPEED
    spawn_gap: float = settings.BASE_SPAWN_GAP
    obstacle_count: int = 20
    wave_size: int = 1


@dataclass(frozen=True)
class GameMode:
    """A sequence of levels plus the road limits the car and obstacles use.

    Attributes:
        name: Human-readable name of the mode.
        levels: The levels, played in order.
        car_bounds: Exclusive ``(left, right)`` limits for the car's ``x``.
        obstacle_x_range: Inclusive range for randomly placed obstacles.
        start_position: Initial ``(x, y)`` of the car.
        loop: Restart from the first level instead of ending after the last.
    """

    name: str
    levels: tuple[LevelSpec, ...]
    car_bounds: tuple[int, int]
    obstacle_x_range: tuple[int, int]
    start_position: tuple[float, float]
    loop: bool = False


_W = settings.WINDOW_WIDTH
_H = settings.WINDOW_HEIGHT
_FAST_GAP = 1.5 * settings.BASE_SPAWN_GAP

#: The five-level campaign for human players.
PLAYER_MODE = GameMode(
    name="player",
    levels=(
        LevelSpec(Pattern.RANDOM),
        LevelSpec(Pattern.RANDOM, speed=40, spawn_gap=_FAST_GAP),
        LevelSpec(Pattern.LANES, speed=35, spawn_gap=_FAST_GAP, wave_size=2),
        LevelSpec(Pattern.WIDE_TRUCK, speed=30, spawn_gap=_FAST_GAP),
        LevelSpec(Pattern.SLALOM, speed=30, spawn_gap=_FAST_GAP, wave_size=2),
    ),
    car_bounds=(25, _W - 95),
    obstacle_x_range=(25, _W - 125),
    start_position=(250, 750),
)

#: The endless two-level course the neural network was trained on.
TRAINING_MODE = GameMode(
    name="training",
    levels=(
        LevelSpec(Pattern.RANDOM),
        LevelSpec(Pattern.RANDOM, speed=40, spawn_gap=_FAST_GAP),
    ),
    car_bounds=(25, _W - 100),
    obstacle_x_range=(25, _W - 100),
    start_position=(_W / 2 - settings.CAR_SIZE[0] // 2, _H - 150),
    loop=True,
)

#: The course used when the trained network drives: a longer fast level.
AI_MODE = replace(
    TRAINING_MODE,
    name="ai",
    levels=(
        TRAINING_MODE.levels[0],
        replace(TRAINING_MODE.levels[1], obstacle_count=100),
    ),
)


def build_wave(
    spec: LevelSpec,
    wave_index: int,
    sprites: Sprites,
    x_range: tuple[int, int],
    rng: random.Random,
) -> list[Obstacle]:
    """Create the obstacles of one wave, placed at the top of the screen.

    Args:
        spec: The level being played.
        wave_index: Number of waves already spawned in this level.
        sprites: The loaded game sprites.
        x_range: Inclusive horizontal range for random placement.
        rng: Random number generator used for placement.
    """
    speed = spec.speed

    def truck_or_van(x: float) -> Obstacle:
        image = sprites.van if rng.randint(0, 1) else sprites.truck
        return Obstacle(x, 0, image, speed)

    match spec.pattern:
        case Pattern.RANDOM:
            return [truck_or_van(rng.randint(*x_range)) for _ in range(spec.wave_size)]
        case Pattern.LANES:
            lanes = np.linspace(x_range[0], x_range[1], 5)
            return [
                truck_or_van(float(lanes[rng.randint(0, 4)]))
                for _ in range(spec.wave_size)
            ]
        case Pattern.WIDE_TRUCK:
            x = 50 if rng.randint(0, 1) else 170
            return [Obstacle(x, 0, sprites.big_truck, speed)]
        case Pattern.SLALOM:
            if wave_index % 2 == 0:
                return [
                    Obstacle(250, 0, sprites.short_big_truck, speed),
                    Obstacle(50, 0, sprites.bus, speed),
                ]
            return [
                Obstacle(
                    50, 0, pygame.transform.rotate(sprites.short_big_truck, 180), speed
                ),
                Obstacle(_W - 150, 0, pygame.transform.rotate(sprites.bus, 180), speed),
            ]
    raise ValueError(f"Unknown pattern: {spec.pattern}")


class ObstacleCourse:
    """Spawns, moves and retires obstacles according to a :class:`GameMode`.

    Args:
        mode: The levels and limits to play.
        sprites: The loaded game sprites.
        rng: Random number generator; pass a seeded one for reproducible runs.
    """

    def __init__(
        self, mode: GameMode, sprites: Sprites, rng: random.Random | None = None
    ) -> None:
        self.mode = mode
        self.sprites = sprites
        self.rng = rng or random.Random()
        self.level_index = 0
        self.obstacles: list[Obstacle] = []
        self.completed = False
        self._start_level()

    @property
    def level(self) -> int:
        """The current level number, starting at 1."""
        return self.level_index + 1

    @property
    def spec(self) -> LevelSpec:
        """The rules of the current level."""
        return self.mode.levels[self.level_index]

    @property
    def nearest(self) -> Obstacle | None:
        """The oldest obstacle still on screen (the one closest to the car)."""
        return self.obstacles[0] if self.obstacles else None

    def _start_level(self) -> None:
        self._spawned = 0
        self._waves = 0
        self._spawn_wave()

    def _spawn_wave(self) -> None:
        wave = build_wave(
            self.spec, self._waves, self.sprites, self.mode.obstacle_x_range, self.rng
        )
        self.obstacles.extend(wave)
        self._spawned += len(wave)
        self._waves += 1

    def spawn_if_ready(self) -> None:
        """Spawn the next wave once the latest one has moved far enough."""
        if (
            self.obstacles
            and self.obstacles[-1].y > self.spec.spawn_gap
            and self._spawned < self.spec.obstacle_count
        ):
            self._spawn_wave()

    def remove_passed(self) -> int:
        """Remove obstacles that left the screen and return how many did."""
        remaining = [o for o in self.obstacles if not o.has_left_screen()]
        passed = len(self.obstacles) - len(remaining)
        self.obstacles = remaining
        return passed

    def advance_level_if_cleared(self) -> bool:
        """Start the next level when the road is empty.

        Returns:
            ``True`` if a level was cleared this call. When the last level of
            a non-looping mode is cleared, :attr:`completed` becomes ``True``
            and no further obstacles spawn.
        """
        if self.obstacles or self.completed:
            return False
        if self.level_index + 1 < len(self.mode.levels):
            self.level_index += 1
        elif self.mode.loop:
            self.level_index = 0
        else:
            self.completed = True
            return True
        self._start_level()
        return True
