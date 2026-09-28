"""Opening the game window and drawing the heads-up display."""

from __future__ import annotations

import os

import pygame

from . import settings
from .assets import load_icon


def open_window(headless: bool = False) -> pygame.Surface:
    """Initialise pygame and open the game window.

    Args:
        headless: Use SDL's dummy video driver so nothing is shown on screen.
            This allows fast training without a display.
    """
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_caption(settings.TITLE)
    pygame.display.set_icon(load_icon())
    return pygame.display.set_mode(settings.WINDOW_SIZE)


def close_window() -> None:
    """Shut pygame down again."""
    pygame.quit()


def quit_requested() -> bool:
    """Drain the event queue and return whether the window was closed."""
    return any(event.type == pygame.QUIT for event in pygame.event.get())


class Hud:
    """Renders the score, level and training statistics on top of the game."""

    def __init__(self) -> None:
        pygame.font.init()
        self.font = pygame.font.SysFont(settings.HUD_FONT, settings.HUD_FONT_SIZE)
        self.small_font = pygame.font.SysFont(
            settings.HUD_FONT, settings.HUD_SMALL_FONT_SIZE
        )

    def draw_score(self, surface: pygame.Surface, score: int, level: int) -> None:
        """Draw the level in the top-left and the score in the top-right."""
        score_label = self.font.render(f"SCORE : {score}", True, settings.WHITE)
        surface.blit(
            score_label,
            (settings.WINDOW_WIDTH - score_label.get_width() - 15, 10),
        )
        level_label = self.font.render(f"LEVEL : {level}", True, settings.WHITE)
        surface.blit(level_label, (15, 10))

    def draw_training(
        self,
        surface: pygame.Surface,
        generation: int,
        alive: int,
        fitness: float,
        best_fitness: float,
    ) -> None:
        """Draw the generation, surviving cars and fitness while training."""
        lines = [
            (self.font, f"GEN : {generation}", (10, 50)),
            (self.font, f"ALIVE : {alive}", (10, 90)),
        ]
        right = settings.WINDOW_WIDTH - 15
        for font, text, position in lines:
            surface.blit(font.render(text, True, settings.ORANGE), position)
        for text, y in (
            (f"best fitness so far : {best_fitness:.2f}", 60),
            (f"fitness of best car : {fitness:.2f}", 80),
        ):
            label = self.small_font.render(text, True, settings.ORANGE)
            surface.blit(label, (right - label.get_width(), y))
