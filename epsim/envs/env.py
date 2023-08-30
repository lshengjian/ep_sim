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
        args: "DictConfig"  = None
    ): # noqa: F821
        
        self.world=World(args.max_x)


        self.cur_crane_index=0
        rows=args.max_x/args.cols+1+2
        self.renderer=Renderer(self.world,args.fps,rows,args.cols,args.tile_size)
        self.observation_space = spaces.Discrete(10) #todo
        self.action_space = spaces.Discrete(5) #todo
        self.render_mode = render_mode
        
        for p in args.products:
            self.world.products[p.code]=[p.num,0] #total,doing



    def next_crane(self):
        self.cur_crane_index=(self.cur_crane_index+1)%len(self.world.all_cranes)
        
    def step(self, a:Actions):
        #print('-'*8)
        crane=self.world.all_cranes[self.cur_crane_index]
        #print(crane,end=' ===> ')
        crane.set_command(a)
        self.world.update()
        #print(crane)
        mask=self.world.mask_one_crane(self.cur_crane_index)

        if self.render_mode == "human":
            self.render()
        return (0, self.world.reward, self.world.is_over, False, {"action_mask": mask})
    
    @property
    def reward(self):
        return self.world.reward
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.cur_crane_index=0
        self.world.reset()
        mask=self.world.mask_one_crane(self.cur_crane_index)
        
        
        if self.render_mode == "human":
            self.render()
        return 0, {"action_mask": mask}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
            )
            return
        
        return self.renderer.render(self.render_mode)
    
    def close(self):
        self.renderer.close()

 