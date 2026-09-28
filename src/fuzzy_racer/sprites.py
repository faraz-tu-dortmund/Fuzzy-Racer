"""The moving objects of the game: the player's car, obstacles and the road."""

from __future__ import annotations

from enum import Enum

import pygame

from . import settings
from .assets import Sprites


class Steer(Enum):
    """A steering command for one frame."""

    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"


class Car:
    """A car that moves sideways within ``bounds`` and can crash into obstacles.

    The car keeps its ``y`` position fixed until it crashes; from then on it
    is dragged along by the obstacle it hit and drawn with an explosion.

    Args:
        x: Initial horizontal position of the sprite's left edge.
        y: Vertical position of the sprite's top edge.
        sprites: The loaded game sprites.
        bounds: Exclusive ``(left, right)`` limits for ``x`` while steering.
    """

    def __init__(
        self, x: float, y: float, sprites: Sprites, bounds: tuple[int, int]
    ) -> None:
        self.x = x
        self.y = y
        self.bounds = bounds
        self.crashed = False
        self._images = {
            Steer.STRAIGHT: pygame.transform.rotate(sprites.car, 0),
            Steer.LEFT: pygame.transform.rotate(sprites.car, settings.CAR_TILT_ANGLE),
            Steer.RIGHT: pygame.transform.rotate(sprites.car, -settings.CAR_TILT_ANGLE),
        }
        # Masks are cached per tilt so collision checks stay cheap even when
        # hundreds of cars race at once during training.
        self._masks = {
            steer: pygame.mask.from_surface(image)
            for steer, image in self._images.items()
        }
        self._blasts = (sprites.small_blast, sprites.large_blast)
        self.pose = Steer.STRAIGHT

    @property
    def image(self) -> pygame.Surface:
        """The sprite for the car's current tilt."""
        return self._images[self.pose]

    @property
    def mask(self) -> pygame.mask.Mask:
        """The collision mask for the car's current tilt."""
        return self._masks[self.pose]

    def can_move(self, steer: Steer) -> bool:
        """Return whether one step in direction ``steer`` stays on the road."""
        left, right = self.bounds
        if steer is Steer.LEFT:
            return self.x - settings.CAR_STEP > left
        if steer is Steer.RIGHT:
            return self.x + settings.CAR_STEP < right
        return True

    def steer(self, steer: Steer) -> None:
        """Tilt the car and move it one step, unless that leaves the road."""
        if steer is Steer.STRAIGHT or not self.can_move(steer):
            self.pose = Steer.STRAIGHT
            return
        self.pose = steer
        step = settings.CAR_STEP if steer is Steer.RIGHT else -settings.CAR_STEP
        self.x += step

    def crash_into(self, obstacle: Obstacle) -> None:
        """Mark the car as crashed and attach it to ``obstacle``."""
        self.crashed = True
        self.y = obstacle.y + settings.CRASH_OFFSET

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the car, plus an explosion if it crashed."""
        surface.blit(self.image, (self.x, self.y))
        if self.crashed:
            for blast in self._blasts:
                surface.blit(blast, (self.x, self.y))


class Obstacle:
    """A vehicle driving down the screen at a constant speed.

    Args:
        x: Horizontal position of the sprite's left edge.
        y: Vertical position of the sprite's top edge.
        image: The sprite to draw; it also defines the collision shape.
        speed: Pixels moved down per frame.
    """

    def __init__(self, x: float, y: float, image: pygame.Surface, speed: int) -> None:
        self.x = x
        self.y = y
        self.image = image
        self.speed = speed
        self.mask = pygame.mask.from_surface(image)

    def move(self) -> None:
        """Move one frame down the screen."""
        self.y += self.speed

    def collides_with(self, car: Car) -> bool:
        """Return whether this obstacle's pixels overlap the car's pixels."""
        offset = (round(self.x) - round(car.x), round(self.y - car.y))
        return car.mask.overlap(self.mask, offset) is not None

    def has_left_screen(self) -> bool:
        """Return whether the obstacle has driven past the bottom edge."""
        return self.y > settings.WINDOW_HEIGHT

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the obstacle."""
        surface.blit(self.image, (self.x, self.y))


class Road:
    """An endlessly scrolling background made of two stacked copies of a tile."""

    def __init__(self, image: pygame.Surface, speed: int = settings.ROAD_SPEED):
        self.image = image
        self.speed = speed
        self.height = image.get_height()
        self.y1 = 0
        self.y2 = self.height

    def scroll(self) -> None:
        """Advance the road by one frame, wrapping tiles that left the screen."""
        self.y1 += self.speed
        self.y2 += self.speed
        if self.y1 - self.height > 0:
            self.y1 = self.y2 - self.height
        if self.y2 - self.height > 0:
            self.y2 = self.y1 - self.height

    def draw(self, surface: pygame.Surface) -> None:
        """Draw both road tiles."""
        surface.blit(self.image, (0, self.y1))
        surface.blit(self.image, (0, self.y2))
