from __future__ import annotations

from typing import  Tuple

import numpy as np
from epsim.core.comm import *
from epsim.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from epsim.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)


Point = Tuple[int, int]


class WorldObj:

    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self) -> bool:
        """Can the agent pick this up?"""
        return False

    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return False



    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty":
            return None


        if obj_type == "workpiece":
            v = Workpiece(color)
        elif obj_type == "agent":
            v = Agent(color)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "tank":
            v = Tank()
        else:
            assert False, f'unknown object type in decode {obj_type:%s}'

        return v

    def render(self, r: np.ndarray) -> np.ndarray:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])




class Tank(WorldObj):
    def __init__(self,  color: str ="blue",is_free: bool = True):
        super().__init__("tank",color)
        self.is_free = is_free

    def can_overlap(self):
        """The agent can only walk over this cell when the tank is free"""
        return self.is_free


    # def toggle(self, env, pos):
    #     # If the player has the right key to open the door
    #     if self.is_locked:
    #         if isinstance(env.carrying, Key) and env.carrying.color == self.color:
    #             self.is_locked = False
    #             self.is_free = True
    #             return True
    #         return False

    #     self.is_free = not self.is_free
    #     return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_free:
            state = StateTank.idle
        else :
            state = StateTank.working
        

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))
        c = COLORS[self.color]

        if not self.is_free:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            

        # # Door frame and door
        # if self.is_locked:
        #     fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        #     fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

        #     # Draw key slot
        #     fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        # else:
        #     fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        #     fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        #     fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
        #     fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

        #     # Draw door handle
        #     fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)





class Workpiece(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("workpiece", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Agent(WorldObj):
    def __init__(self, color="red"):
        super().__init__("agent", color)


    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.4), COLORS[self.color])