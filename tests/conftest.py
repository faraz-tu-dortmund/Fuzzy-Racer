import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame  # noqa: E402
import pytest  # noqa: E402

from fuzzy_racer.assets import load_sprites  # noqa: E402


@pytest.fixture(scope="session")
def sprites():
    pygame.init()
    pygame.display.set_mode((1, 1))
    yield load_sprites()
    pygame.quit()
