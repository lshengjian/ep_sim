from __future__ import annotations

import numpy as np

from enum import IntEnum


class Actions(IntEnum):
    stay = 0
    forward = 1
    back = 2
    up = 3
    down = 4

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5,"white":6}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    "empty": 0,
    "start": 1,
    "tank": 2,
    "exchange": 3,
    "end": 4,
        
    "workpiece": 5,
    "crane": 6,
    
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


# Map of agent direction indices to vectors
# DIR_TO_VEC = [
#     # Pointing right (positive X)
#     np.array((1, 0)),
#     # Down (positive Y)
#     np.array((0, 1)),
#     # Pointing left (negative X)
#     np.array((-1, 0)),
#     # Up (negative Y)
#     np.array((0, -1)),
# ]
