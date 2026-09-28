"""Evolving a car-driving neural network with NEAT.

Each generation, every genome drives its own car on the same obstacle course.
Fitness rewards survival and penalises crashes:

* +0.1 for every frame the car is on the road,
* -2 for every frame (and obstacle) the car is touching an obstacle,
* +1 for every obstacle that passes while the car is still intact,
* +5 for every level the car clears.

Crashed cars are removed once the next obstacle leaves the screen. The
generation ends when every car is gone or the best car reaches the fitness
threshold from the NEAT config, which also ends training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import neat
import pygame

from . import settings
from .assets import NEAT_CONFIG_PATH, load_sprites
from .display import Hud, close_window, open_window, quit_requested
from .levels import TRAINING_MODE, GameMode
from .network import Network, steer_from_output
from .race import Race
from .visualize import plot_network

ALIVE_REWARD = 0.1
COLLISION_PENALTY = 2.0
PASS_REWARD = 1.0
LEVEL_REWARD = 5.0


class TrainingInterruptedError(Exception):
    """Raised when the training window is closed."""


def load_config(path: str | Path | None = None) -> neat.Config:
    """Load a NEAT config file (the bundled one by default)."""
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path or NEAT_CONFIG_PATH),
    )


class GenerationEvaluator:
    """Callable passed to :meth:`neat.Population.run` to score a generation.

    Args:
        surface: The window to draw on.
        mode: The course the population races on.
        headless: Skip drawing and frame limiting for fast training.
        seed: Seed for obstacle placement; generation ``n`` uses ``seed + n``.
        max_frames: Optional cap on frames per generation.
    """

    def __init__(
        self,
        surface: pygame.Surface,
        mode: GameMode = TRAINING_MODE,
        *,
        headless: bool = False,
        seed: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.surface = surface
        self.mode = mode
        self.headless = headless
        self.seed = seed
        self.max_frames = max_frames
        self.sprites = load_sprites()
        self.hud = None if headless else Hud()
        self.clock = pygame.time.Clock()
        self.generation = 0
        self.best_fitness = float("-inf")

    def __call__(self, genomes: list[tuple[int, neat.DefaultGenome]], config) -> None:
        """Race every genome and store its fitness on ``genome.fitness``."""
        seed = None if self.seed is None else self.seed + self.generation
        race = Race(self.mode, self.sprites, random.Random(seed))
        drivers = []
        for _, genome in genomes:
            genome.fitness = 0.0
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            drivers.append((race.new_car(), network, genome))

        threshold = config.fitness_threshold
        score = frames = 0
        while drivers:
            if not self.headless:
                self.clock.tick(settings.FPS)
            if quit_requested():
                raise TrainingInterruptedError

            for car, network, genome in drivers:
                genome.fitness += ALIVE_REWARD
                output = network.activate(race.sensors(car))[0]
                car.steer(steer_from_output(output))

            cars = [car for car, _, _ in drivers]
            frame = race.step(cars)
            frames += 1
            for index in frame.collisions:
                drivers[index][2].fitness -= COLLISION_PENALTY
            for _ in range(frame.passed):
                drivers = [d for d in drivers if not d[0].crashed]
                score += 1
                for _, _, genome in drivers:
                    genome.fitness += PASS_REWARD
            if frame.level_cleared:
                for _, _, genome in drivers:
                    genome.fitness += LEVEL_REWARD

            leader = max((g.fitness for _, _, g in drivers), default=None)
            if leader is not None:
                self.best_fitness = max(self.best_fitness, leader)
                if self.hud is not None:
                    self._draw(race, drivers, score, leader)
                if leader > threshold:
                    break
            if race.completed or (
                self.max_frames is not None and frames >= self.max_frames
            ):
                break
        self.generation += 1

    def _draw(self, race: Race, drivers: list, score: int, leader: float) -> None:
        race.draw(self.surface, [car for car, _, _ in drivers])
        self.hud.draw_score(self.surface, score, race.level)
        self.hud.draw_training(
            self.surface, self.generation, len(drivers), leader, self.best_fitness
        )
        pygame.display.update()


@dataclass(frozen=True)
class TrainingResult:
    """What a training run produced.

    Attributes:
        network: The best network found.
        fitness: Fitness of the best genome.
        generations: Number of completed generations.
        stopped_early: Whether training was interrupted by the user.
        files: Paths of all files written to the output directory.
    """

    network: Network
    fitness: float
    generations: int
    stopped_early: bool
    files: dict[str, Path]

    def __str__(self) -> str:
        status = "stopped early" if self.stopped_early else "finished"
        lines = [
            f"Training {status} after {self.generations} generations, "
            f"best fitness {self.fitness:.2f}"
        ]
        lines += [f"  {name}: {path}" for name, path in self.files.items()]
        return "\n".join(lines)


def train(
    generations: int = 1000,
    *,
    config_path: str | Path | None = None,
    mode: GameMode = TRAINING_MODE,
    output_dir: str | Path = "results/training",
    headless: bool = False,
    seed: int | None = None,
    max_frames: int | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Evolve a driving network and save it together with training plots.

    Closing the window (or pressing Ctrl+C) stops training early; the best
    genome found so far is still saved.

    Args:
        generations: Maximum number of generations to run.
        config_path: NEAT config file; the bundled one is used by default.
        mode: The course to train on.
        output_dir: Directory for the network JSON and the plots.
        headless: Train without a window, as fast as possible.
        seed: Seed for obstacle placement and NEAT's random mutations.
        max_frames: Optional cap on frames per generation.
        verbose: Print NEAT's progress report to the terminal.

    Returns:
        The best network and the paths of the written files:
        ``network`` (JSON) and ``network_plot`` (PNG).
    """
    if seed is not None:
        random.seed(seed)
    config = load_config(config_path)
    population = neat.Population(config)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    if verbose:
        population.add_reporter(neat.StdOutReporter(True))

    surface = open_window(headless)
    evaluator = GenerationEvaluator(
        surface, mode, headless=headless, seed=seed, max_frames=max_frames
    )
    stopped_early = False
    try:
        population.run(evaluator, generations)
    except (TrainingInterruptedError, KeyboardInterrupt):
        stopped_early = True
    finally:
        close_window()

    if not stats.most_fit_genomes:
        raise RuntimeError("Training stopped before the first generation finished.")
    best = stats.best_genome()
    network = Network.from_genome(best, config)

    output_dir = Path(output_dir)
    files = {
        "network": network.save(output_dir / "best_network.json"),
        "network_plot": plot_network(network, output_dir / "best_network.png"),
    }
    return TrainingResult(
        network=network,
        fitness=best.fitness,
        generations=len(stats.most_fit_genomes),
        stopped_early=stopped_early,
        files=files,
    )
