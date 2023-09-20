from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.utils import seeding
import random
from ..core import *
from ..core.componets import Color
from ..core.world_object import WorldObj
from ..core.slot import Slot
from ..core.world import World
from epsim.utils import *
from .render.renderer import Renderer
class MyEnv(Env):

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }
    @property 
    def one_observation_size(self):
        return SHARE.OBJ_TYPE_SIZE+SHARE.OP_TYPE1_SIZE+SHARE.OP_TYPE2_SIZE+SHARE.PRODUCT_TYPE_SIZE+4
    def __init__(
        self,
        render_mode: Optional[str] = "human",
        args: "DictConfig"  = None
    ): # noqa: F821
        
        self.args=args
        MyEnv.metadata['render_fps']=args.fps
        Renderer.LANG=args.language
        SHARE.LOG_LEVEL=args.log_level
        SHARE.TILE_SIZE=args.tile_size
        SHARE.SHORT_ALARM_TIME=args.alarm.short_time
        SHARE.LONG_ALARM_TIME=args.alarm.long_time
        #SHARE.AUTO_DISPATCH=args.auto_dispatch
        SHARE.OBSERVATION_IMAGE=args.observation_image
        self.world=World(args.data_directory,args.max_steps,args.auto_put_starts,args.auto_dispatch_crane)


        ncols=args.screen_columns
        max_x=max(list(self.world.pos_slots.keys()))
        SHARE.MAX_X=max_x
        
        nrows=max_x//ncols+2

        self.renderer=Renderer(self.world,args.fps,nrows,ncols)
        self.render_mode = render_mode
        self.action_space=spaces.Discrete(3)
        self.observation_space=spaces.Box(-1,1,(SHARE.MAX_STATE_LIST_LEN*self.one_observation_size,),dtype=np.float32)

    # def next_crane(self):
    #     for c in self.world.all_cranes:
    #         c.color=Color(255,255,255)
    #     self.world.next_crane()
    #     self.world.cur_crane.color=Color(255,0,0)

    def _make_jobs(self):
        ps=[]
        for p in self.args.products[self.args.data_directory]:
            ps.extend([p.code]*p.num)
        #print(ps)
        self.world.add_jobs(ps)

    def _get_name_by_id(self,id):
        for agv in self.world.all_cranes:
            if agv.id==id:
                return agv.cfg.name
        return None
    def _decision(self):
        masks=self.world.masks
        actions=[0]*len(self.world.all_cranes)
        for idx,mask in enumerate(masks.values()):
            if sum(mask)>1:
                mask[CraneAction.stay]=0
           
            acts=np.argwhere(mask).ravel()
            actions[idx]=random.choice(acts)
        return actions       
    
    def _make_mask(self):
        for agv in self.world.all_cranes:
            self.world.get_masks(agv)
    
    def step(self, a:DispatchAction):
        self.world._reward=0 
        self.world.do_dispatch(a)
        if self.world.auto_dispatch_crane:
            actions=self._decision()
            self.world.set_crane_commands(actions)
        self.world.update()

        if self.render_mode!=None and self.render_mode!='ansi':
            self.render()
        obs=self.world.get_state().ravel()
        self._make_mask()
        return obs,self.world.reward, self.world.is_over, self.world.is_timeout, {}


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.world.reset()
        self._make_jobs()
        if self.render_mode!=None and self.render_mode!='ansi':
            self.render()
        obs=self.world.get_state().ravel()
        self._make_mask()
        return obs, {}



    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.screen_img=self.renderer.render(self.render_mode)
        return self.screen_img

    def close(self):
        self.renderer.close()