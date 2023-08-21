from __future__ import annotations

from epsim.core.constants import COLOR_NAMES
from epsim.core.grid import Grid

from epsim.core.world_object import Workpiece, Tank, Agent,Goal
from .minigrid_env import MiniGridEnv


class PlaygroundEnv(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.
    """

    def __init__(self, max_steps=100, **kwargs):
        self.size = 19
        super().__init__(
            width=self.size,
            height=3,
            max_steps=max_steps,
            **kwargs,
        )



    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        roomW = width // 3
        roomH = height // 3


        # Randomize the player start position and orientation
        self.place_agent()

        # Place random objects in the world
        types = ["workpiece", "tank"]
        for i in range(0, 12):
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)
            if objType == "workpiece":
                obj = Workpiece(objColor)
            elif objType == "tank":
                obj = Tank(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key, ball and box.".format(
                        objType
                    )
                )
            self.place_obj(obj)

        # No explicit mission in this environment
        self.goal = self.place_obj(Goal)
