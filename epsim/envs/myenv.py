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

        SHARE.TILE_SIZE=args.tile_size
        SHARE.SHORT_ALARM_TIME=args.alarm.short_time
        SHARE.LONG_ALARM_TIME=args.alarm.long_time

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
        if SHARE.OBSERVATION_IMAGE:
            tsize=SHARE.TILE_SIZE
            self.observation_space=spaces.Box(0,255,(3*tsize,SHARE.MAX_STATE_LIST_LEN*tsize,3),dtype=np.uint8) #todo



    def _make_jobs(self):
        ps=[]
        for p in self.args.products[self.args.data_directory]:
            ps.extend([p.code]*p.num)
        self.world.add_jobs(ps)

    def _get_name_by_id(self,id):
        for agv in self.world.all_cranes:
            if agv.id==id:
                return agv.cfg.name
        return None
   
    
    def _make_mask(self):
        self.world.get_dispatch_masks()
        for agv in self.world.all_cranes:
            self.world.get_masks(agv)

    def decision(self,infos:dict):
        actions={}
        for k,info in infos.items():
            #print(k,info)
            masks=info['action_masks']
            if k==SHARE.DISPATCH_CODE:
                continue
            t=sum(masks)

            if t>1:
                masks[0]=0
            elif t==0:
                masks[0]=1
                #print(k,masks.tolist())
            
            acts=np.argwhere(masks).ravel()
            actions[k]=random.choice(acts)
        return actions    
    def step(self, a:DispatchAction):

        self.world.do_dispatch(a)
        if self.world.auto_dispatch_crane:
            actions=self.decision(self.world.masks)
            #actions=map(lambda k:self.m actions )
            self.world.set_crane_commands(actions)
        self.world.update()

        if self.render_mode!=None and self.render_mode!='ansi':
            self.render()
        obs=self.world.get_state().ravel()
        self._make_mask()
        if SHARE.OBSERVATION_IMAGE:
            obs=self.world.get_state_img(self.screen_img,self.renderer.nrows,self.renderer.ncols)
        return obs,self.world.reward, self.world.is_over, self.world.is_timeout, self.world.masks


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.world.reset()
        self._make_jobs()
        obs=self.world.get_state().ravel()
        if self.render_mode!=None and self.render_mode!='ansi':
            self.render()
        if SHARE.OBSERVATION_IMAGE:
            obs=self.world.get_state_img(self.screen_img,self.renderer.nrows,self.renderer.ncols)

        self._make_mask()
        return obs,self.world.masks



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