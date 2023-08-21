# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    idle = 0
    forward = 1
    back = 2
    up = 3
    down = 4

class StateTank(IntEnum):
    idle = 0
    working = 1
    alarm = 2

class StateHandCrane(IntEnum):
    stop = 0
    forwarding = 1
    backing = 2
    uping = 3
    downing = 4