import random

import pytest

from fuzzy_racer import AI_MODE, PLAYER_MODE, ObstacleCourse, Pattern, Race
from fuzzy_racer.levels import LevelSpec, build_wave


def drain(course):
    """Run the course with no car until the current level is cleared."""
    level = course.level
    while course.level == level and not course.completed:
        course.spawn_if_ready()
        for obstacle in course.obstacles:
            obstacle.move()
        course.remove_passed()
        course.advance_level_if_cleared()


def test_player_mode_levels_then_completes(sprites):
    course = ObstacleCourse(PLAYER_MODE, sprites, random.Random(1))
    for expected in range(1, 6):
        assert course.level == expected
        drain(course)
    assert course.completed


def test_ai_mode_loops(sprites):
    course = ObstacleCourse(AI_MODE, sprites, random.Random(1))
    drain(course)
    assert course.level == 2
    drain(course)
    assert course.level == 1 and not course.completed


def test_level_sends_obstacle_count(sprites):
    spec = LevelSpec(Pattern.RANDOM, obstacle_count=7)
    mode = PLAYER_MODE.__class__(**{**PLAYER_MODE.__dict__, "levels": (spec,)})
    course = ObstacleCourse(mode, sprites, random.Random(0))
    passed = 0
    while not course.completed:
        course.spawn_if_ready()
        for obstacle in course.obstacles:
            obstacle.move()
        passed += course.remove_passed()
        course.advance_level_if_cleared()
    assert passed == 7


@pytest.mark.parametrize("pattern", list(Pattern))
def test_every_pattern_builds_a_wave(sprites, pattern):
    spec = LevelSpec(pattern, wave_size=2)
    for wave_index in range(2):
        wave = build_wave(spec, wave_index, sprites, (25, 475), random.Random(0))
        assert wave and all(o.y == 0 for o in wave)


def test_race_sensors(sprites):
    race = Race(AI_MODE, sprites, random.Random(0))
    car = race.new_car()
    x, gap, obstacle_x, obstacle_y, level = race.sensors(car)
    assert x == 265 and gap == 600 - 265
    assert obstacle_x == race.course.nearest.x and obstacle_y == 0
    assert level == 1
