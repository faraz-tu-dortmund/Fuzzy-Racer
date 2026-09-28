from fuzzy_racer import Car, Obstacle, Road, Steer
from fuzzy_racer.settings import CAR_STEP, WINDOW_HEIGHT


def test_car_steers_within_bounds(sprites):
    car = Car(100, 750, sprites, bounds=(25, 505))
    car.steer(Steer.LEFT)
    assert car.x == 100 - CAR_STEP
    assert car.pose is Steer.LEFT
    for _ in range(20):
        car.steer(Steer.LEFT)
    assert car.x > 25
    assert car.pose is Steer.STRAIGHT  # no tilt when blocked by the road edge


def test_obstacle_collision_and_crash(sprites):
    car = Car(250, 750, sprites, bounds=(25, 505))
    hit = Obstacle(250, 700, sprites.truck, speed=20)
    miss = Obstacle(25, 700, sprites.truck, speed=20)
    assert hit.collides_with(car)
    assert not miss.collides_with(car)
    car.crash_into(hit)
    assert car.crashed and car.y == 700 + 60


def test_obstacle_leaves_screen(sprites):
    obstacle = Obstacle(0, WINDOW_HEIGHT - 10, sprites.truck, speed=20)
    assert not obstacle.has_left_screen()
    obstacle.move()
    assert obstacle.has_left_screen()


def test_road_wraps(sprites):
    road = Road(sprites.road)
    for _ in range(1000):
        road.scroll()
        assert -road.height <= min(road.y1, road.y2) <= 0
        assert abs(road.y1 - road.y2) == road.height
