from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from ..core import Actions
from ..core.world import World
from .renderer import Renderer
class MyGrid(Env):
    """
    ## Action Space
    The action shape is `(1,)` in the range `{0, 4}` indicating
    which direction to move the player.

    - 0: Stay
    - 1: Move left
    - 2: Move down
    - 3: Move right
    - 4: Move up

    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * nrows + current_col (where both the row and col start at 0).

    ## Rewards

    Reward schedule:
    - Reach goal: +1
    - Reach forbidden: -1
    - Reach frozen: 0

    ## Episode End
    The episode ends when The player reaches the goal 

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = "human",
        fps=4,
    ):
        self.world=World()
        self.renderer=Renderer(self.world,fps)
        self.observation_space = spaces.Discrete(10) #todo
        self.action_space = spaces.Discrete(5) #todo
        self.render_mode = render_mode


    # def action_mask(self, state: int):
    #     mask = np.ones(self.world.nA, dtype=np.int8)
    #     nrow=self.world.nrow
    #     ncol=self.world.ncol
    #     row,col=self.world.state2idx(state)
    #     if col==0:
    #         mask[LEFT]=0
    #     elif col==ncol-1:
    #         mask[RIGHT]=0
    #     if row==0:
    #         mask[UP]=0
    #     elif row==nrow-1:
    #         mask[DOWN]=0
    #     return mask
    
    def step(self, a:Actions):
        #s=self.world.state
        self.world.all_cranes[0].set_command(a)
        self.world.update()

        if self.render_mode == "human":
            self.render()
        return (0, self.world.reward, self.world.is_over, False, {"action_mask": None})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        #self.lastaction = None
        if self.render_mode == "human":
            self.render()
        return 0, {"action_mask": None}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
            )
            return
        
        return self.renderer.render(self.render_mode)

 