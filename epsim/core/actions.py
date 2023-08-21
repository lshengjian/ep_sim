# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    idle = 0
    forward = 1
    back = 2
    up = 3
    down = 4

