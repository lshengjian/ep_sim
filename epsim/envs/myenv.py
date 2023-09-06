from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.utils import seeding

from ..core import Actions
from ..core.componets import Color
from ..core.world_object import WorldObj
from ..core.slot import Slot
from ..core.world import World
from epsim.utils import *
from .render.renderer import Renderer
class MyEnv(Env):
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
        
        self.args=args
        self.world=World(args.config_directory,args.max_x)
        Renderer.LANG=args.language
        WorldObj.TILE_SIZE=args.tile_size
        Slot.WarningTime=args.alarm.warning
        Slot.FatalTime=args.alarm.fatal

        ncols=args.screen_columns
        
        rows=int(args.max_x/ncols+0.5)+1
        self.renderer=Renderer(self.world,args.fps,rows,ncols,args.tile_size)
        self.observation_space = spaces.Box(0,255) #todo
        self.action_space = spaces.Discrete(5) #todo
        self.render_mode = render_mode
        self.machines_img=None
        self.product_img=None

    def next_crane(self):
        for c in self.world.all_cranes:
            c.color=Color(255,255,255)
        self.world.next_crane()
        self.world.cur_crane.color=Color(255,0,0)
        
    def step(self, a:Actions):
        self.world.set_command(a)
        self.world.update()
        #print(crane)
        mask=self.world.mask_action(self.world.cur_crane)
        self.render()
        obs=self.get_observations()
        return (obs, self.world.reward, self.world.is_over, False, {"action_mask": mask})
    
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
        self.world.reset()
        ps=[]
        for p in self.args.products:
            ps.extend([p.code]*p.num)
        self.world.add_jobs(ps)
        for c in self.world.all_cranes:
            c.color=Color(255,255,255)
        self.world.cur_crane.color=Color(255,0,0)
        mask=self.world.mask_action(self.world.cur_crane)
        obs=self.render()
        return obs, {"action_mask": mask}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
            )
            return None
        
        self.state= self.renderer.render(self.render_mode)
        ms,ps=get_state(self.state,len(self.world.products),self.args.screen_columns,self.args.max_x,WorldObj.TILE_SIZE)
        self.state=merge_two_images(ms,ps,WorldObj.TILE_SIZE)
        self.machines_img=merge_images(ms)
        return self.state
    
    def close(self):
        self.renderer.close()

    def get_state(self):# todo
        return self.state
    
    def get_observations(self):
        return get_observation(self.machines_img,self.world.cur_crane.x,13) 