"""The two single-car game modes: a human player or a trained network driving."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import pygame

from . import settings
from .assets import load_sprites
from .display import Hud, close_window, open_window, quit_requested
from .levels import AI_MODE, PLAYER_MODE, GameMode
from .network import Network, load_pretrained_network
from .race import Race
from .sprites import Steer

#: A driver maps the car's sensor readings to a steering command.
Driver = Callable[[tuple[float, ...]], Steer]


@dataclass(frozen=True)
class GameResult:
    """The outcome of one game.

    Attributes:
        score: Obstacles dodged before crashing.
        level: The level reached.
        frames: Number of frames simulated.
        outcome: ``"crashed"``, ``"won"``, ``"quit"`` or ``"timeout"``.
    """

    score: int
    level: int
    frames: int
    outcome: str

    def __str__(self) -> str:
        return (
            f"{self.outcome.capitalize()} on level {self.level} "
            f"with a score of {self.score} ({self.frames} frames)."
        )


def keyboard_driver(_sensors: tuple[float, ...]) -> Steer:
    """Steer with the left and right arrow keys."""
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return Steer.LEFT
    if keys[pygame.K_RIGHT]:
        return Steer.RIGHT
    return Steer.STRAIGHT


def run_game(
    mode: GameMode,
    driver: Driver,
    *,
    headless: bool = False,
    seed: int | None = None,
    max_frames: int | None = None,
) -> GameResult:
    """Play one game with a single car until it crashes or the mode is won.

    A crashed car is dragged along by the obstacle it hit; the game ends once
    that obstacle leaves the screen, so the explosion stays visible.

    Args:
        mode: The levels to play.
        driver: Chooses the steering command every frame.
        headless: Simulate as fast as possible without showing a window.
        seed: Seed for obstacle placement, for reproducible games.
        max_frames: Stop after this many frames (useful for strong networks
            on endless modes).
    """
    surface = open_window(headless)
    try:
        return simulate(surface, mode, driver, headless, seed, max_frames)
    finally:
        close_window()


def simulate(
    surface: pygame.Surface,
    mode: GameMode,
    driver: Driver,
    headless: bool = False,
    seed: int | None = None,
    max_frames: int | None = None,
) -> GameResult:
    """Run the game loop on an already opened window; see :func:`run_game`."""
    race = Race(mode, load_sprites(), random.Random(seed))
    car = race.new_car()
    hud = None if headless else Hud()
    clock = pygame.time.Clock()
    score = frames = 0

    def result(outcome: str) -> GameResult:
        return GameResult(score, race.level, frames, outcome)

    while True:
        if not headless:
            clock.tick(settings.FPS)
        if quit_requested():
            return result("quit")
        car.steer(driver(race.sensors(car)))
        frame = race.step([car])
        frames += 1
        if frame.passed and car.crashed:
            return result("crashed")
        score += frame.passed
        if hud is not None:
            race.draw(surface, [car])
            hud.draw_score(surface, score, race.level)
            pygame.display.update()
        if race.completed:
            return result("won")
        if max_frames is not None and frames >= max_frames:
            return result("timeout")


def play(seed: int | None = None) -> GameResult:
    """Play the five-level campaign with the arrow keys."""
    return run_game(PLAYER_MODE, keyboard_driver, seed=seed)


def watch_ai(
    network: Network | None = None,
    *,
    mode: GameMode = AI_MODE,
    headless: bool = False,
    seed: int | None = None,
    max_frames: int | None = None,
) -> GameResult:
    """Let a trained network drive (the bundled pretrained one by default).

    By default the network plays the endless course it was trained on; pass
    ``mode=PLAYER_MODE`` to see how it copes with the human campaign.
    """
    network = network or load_pretrained_network()
    return run_game(
        mode,
        network.decide,
        headless=headless,
        seed=seed,
        max_frames=max_frames,
    )
