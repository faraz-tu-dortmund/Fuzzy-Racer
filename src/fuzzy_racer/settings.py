"""Game-wide constants shared by every Fuzzy Racer mode."""

TITLE = "Fuzzy Racer"

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 900
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
FPS = 30

#: Pixels the road scrolls per frame.
ROAD_SPEED = 7
#: Obstacle speed (pixels per frame) on the first level.
BASE_OBSTACLE_SPEED = 20
#: A new wave spawns once the previous one travelled this far down the screen.
BASE_SPAWN_GAP = 0.5 * WINDOW_HEIGHT

#: Pixels the car moves sideways per frame while steering.
CAR_STEP = 20
#: Degrees the car sprite tilts while steering.
CAR_TILT_ANGLE = 20
#: A crashed car is dragged along this far below the obstacle's top edge.
CRASH_OFFSET = 60

# Sprite sizes (width, height) in pixels.
CAR_SIZE = (70, 100)
TRUCK_SIZE = (95, 125)
BIG_TRUCK_SIZE = (400, 100)
SHORT_BIG_TRUCK_SIZE = (300, 100)
BUS_SIZE = (100, 100)
SMALL_BLAST_SIZE = (75, 75)
LARGE_BLAST_SIZE = (150, 150)

# HUD colours (RGB).
WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)
#: Font names tried in order; the first one installed is used.
HUD_FONT = "comicsansms,comicsans,dejavusans,freesans,arial"
HUD_FONT_SIZE = 50
HUD_SMALL_FONT_SIZE = 25
