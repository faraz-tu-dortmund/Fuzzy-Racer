"""The simulation core shared by the human game, the AI game and training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pygame

from . import settings
from .assets import Sprites
from .levels import GameMode, ObstacleCourse
from .sprites import Car, Road


@dataclass
class FrameResult:
    """What happened during one simulated frame.

    Attributes:
        passed: Number of obstacles that left the screen.
        collisions: Index (into the ``cars`` passed to :meth:`Race.step`) of
            every car that touched an obstacle, once per obstacle touched.
        level_cleared: Whether the current level was finished.
    """

    passed: int = 0
    collisions: list[int] = field(default_factory=list)
    level_cleared: bool = False


class Race:
    """A road, an obstacle course and the logic that moves them frame by frame.

    Cars are owned by the caller, so the same race can host a single player
    or a whole population of neural networks.

    Args:
        mode: The levels and road limits to play.
        sprites: The loaded game sprites.
        rng: Random number generator; pass a seeded one for reproducible runs.
    """

    def __init__(
        self, mode: GameMode, sprites: Sprites, rng: random.Random | None = None
    ) -> None:
        self.mode = mode
        self.sprites = sprites
        self.road = Road(sprites.road)
        self.course = ObstacleCourse(mode, sprites, rng)

    @property
    def level(self) -> int:
        """The current level number, starting at 1."""
        return self.course.level

    @property
    def completed(self) -> bool:
        """Whether the last level of a non-looping mode was cleared."""
        return self.course.completed

    def new_car(self) -> Car:
        """Create a car at the mode's start position."""
        x, y = self.mode.start_position
        return Car(x, y, self.sprites, self.mode.car_bounds)

    def sensors(self, car: Car) -> tuple[float, float, float, float, int]:
        """Return the neural network inputs describing the race for ``car``.

        The inputs are, in order: the car's ``x``, its distance to the right
        window edge, the nearest obstacle's ``x`` and ``y``, and the level.
        """
        nearest = self.course.nearest
        obstacle_x, obstacle_y = (nearest.x, nearest.y) if nearest else (0.0, 0.0)
        return (
            car.x,
            settings.WINDOW_WIDTH - car.x,
            obstacle_x,
            obstacle_y,
            self.level,
        )

    def step(self, cars: list[Car]) -> FrameResult:
        """Advance the race by one frame.

        Cars should be steered before calling this. Every obstacle is checked
        for collisions with every car and then moved; obstacles that left the
        screen are removed, and the next level starts when the road is empty.
        """
        result = FrameResult()
        self.road.scroll()
        self.course.spawn_if_ready()
        for obstacle in self.course.obstacles:
            for index, car in enumerate(cars):
                if obstacle.collides_with(car):
                    car.crash_into(obstacle)
                    result.collisions.append(index)
            obstacle.move()
        result.passed = self.course.remove_passed()
        result.level_cleared = self.course.advance_level_if_cleared()
        return result

    def draw(self, surface: pygame.Surface, cars: list[Car]) -> None:
        """Draw the road, the cars and the obstacles (without the HUD)."""
        self.road.draw(surface)
        for car in cars:
            car.draw(surface)
        for obstacle in self.course.obstacles:
            obstacle.draw(surface)
