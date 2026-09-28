"""Locating and loading the images and data files bundled with the package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pygame

from . import settings

DATA_DIR = Path(str(resources.files("fuzzy_racer") / "data"))
IMAGE_DIR = DATA_DIR / "imgs"
NEAT_CONFIG_PATH = DATA_DIR / "neat_config.ini"
PRETRAINED_NETWORK_PATH = DATA_DIR / "pretrained_network.json"


def image_path(name: str) -> Path:
    """Return the path of a bundled image, e.g. ``image_path("car.png")``."""
    return IMAGE_DIR / name


def load_image(name: str, size: tuple[int, int]) -> pygame.Surface:
    """Load a bundled image and scale it to ``size``.

    The surface is converted to the display's pixel format when a display
    exists, which makes blitting much faster. Without a display (e.g. in unit
    tests) the raw surface is returned, which still supports collision masks.
    """
    surface = pygame.image.load(str(image_path(name)))
    if pygame.display.get_surface() is not None:
        surface = surface.convert_alpha()
    return pygame.transform.scale(surface, size)


@dataclass(frozen=True)
class Sprites:
    """All surfaces used by the game, loaded once and shared by every sprite."""

    car: pygame.Surface
    truck: pygame.Surface
    van: pygame.Surface
    big_truck: pygame.Surface
    short_big_truck: pygame.Surface
    bus: pygame.Surface
    small_blast: pygame.Surface
    large_blast: pygame.Surface
    road: pygame.Surface


def load_sprites() -> Sprites:
    """Load every game sprite at the size the game uses."""
    big_truck = load_image("big_truck.png", settings.BIG_TRUCK_SIZE)
    return Sprites(
        car=load_image("car.png", settings.CAR_SIZE),
        truck=load_image("truck.png", settings.TRUCK_SIZE),
        van=load_image("random.png", settings.TRUCK_SIZE),
        big_truck=big_truck,
        short_big_truck=pygame.transform.scale(
            big_truck, settings.SHORT_BIG_TRUCK_SIZE
        ),
        bus=load_image("bus.png", settings.BUS_SIZE),
        small_blast=load_image("blast.png", settings.SMALL_BLAST_SIZE),
        large_blast=load_image("blast.png", settings.LARGE_BLAST_SIZE),
        road=load_image("road.png", settings.WINDOW_SIZE),
    )


def load_icon() -> pygame.Surface:
    """Load the window icon, falling back to the PNG if SVG is unsupported."""
    try:
        return pygame.image.load(str(image_path("icon.svg")))
    except pygame.error:
        return pygame.image.load(str(image_path("icon_png.png")))
