# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    idle = 0
    left = 1
    right = 2
    forward = 3

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