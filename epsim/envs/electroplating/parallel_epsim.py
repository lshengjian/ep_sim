import functools
import numpy as np
import gymnasium
from gymnasium.spaces import Discrete,Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from ..render.renderer import Renderer
from epsim.core import World,WorldObj,Slot,Crane,Actions,SHARE
from epsim.core.componets import Color
from epsim.core import SHARE

import logging
logger = logging.getLogger(__name__)
def env(render_mode=None,args: "DictConfig"  = None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(render_mode=render_mode,args=args)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None,args: "DictConfig"  = None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode,args=args)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes":  ["human", "rgb_array"],"name": "electroplating_v1"}

    def __init__(self, render_mode=None,args: "DictConfig"  = None):
        self.args=args
        Renderer.LANG=args.language
        SHARE.LOG_LEVEL=args.log_level
        SHARE.TILE_SIZE=args.tile_size
        SHARE.SHORT_ALARM_TIME=args.alarm.short_time
        SHARE.LONG_ALARM_TIME=args.alarm.long_time
        SHARE.AUTO_DISPATCH=args.auto_dispatch
        SHARE.OBSERVATION_IMAGE=args.observation_image
        self.world=World(args.data_directory,SHARE.AUTO_DISPATCH)


        ncols=args.screen_columns
        max_x=max(list(self.world.pos_slots.keys()))
        SHARE.MAX_X=max_x
        
        nrows=max_x//ncols+2

        self.renderer=Renderer(self.world,args.fps,nrows,ncols)
        
        self.possible_agents = [crane.cfg.name for crane in self.world.all_cranes]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.world.all_cranes))))
        )
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    
    @property 
    def one_observation_size(self):
        return SHARE.OBJ_TYPE_SIZE+SHARE.OP_TYPE1_SIZE+SHARE.OP_TYPE2_SIZE+SHARE.PRODUCT_TYPE_SIZE+4
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):#todo  dispatch see all obs
        rt={'observation':None,'image':None}
        if SHARE.OBSERVATION_IMAGE:
            tsize=SHARE.TILE_SIZE
            rt['image']=Box(0,255,(3*tsize,(2*SHARE.MAX_AGENT_SEE_DISTANCE+1 )*tsize,3),dtype=np.uint8)
        else:
            rt['observation']=Box(-1,1,(SHARE.MAX_OBS_LIST_LEN*self.one_observation_size,),dtype=np.float32)

        
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return rt

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
    
    # def get_state(self):# todo
    #     return self.state
    
    # def get_observations(self):
    #     return get_observation(self.machines_img,self.world.cur_crane.x,13) 

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.screen_img=self.renderer.render(self.render_mode)
        return self.screen_img

    def close(self):
        self.renderer.close()

    # def observe(self, agent: str):
    #     print(f'observe for {agent}')
    #     return self.observations[agent]


    def _make_jobs(self):
        ps=[]
        for p in self.args.products[self.args.data_directory]:
            ps.extend([p.code]*p.num)
        #print(ps)
        self.world.add_jobs(ps)
        for agv in self.world.all_cranes:
            agv.color=Color(255,255,255)
        self.world.cur_crane.color=Color(255,0,0) 

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        nrows=self.renderer.nrows
        ncols=self.renderer.ncols
        self.world.reset()
        self.agents = self.possible_agents[:]
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            screen_img=self.render()
            self.world.get_state_img(screen_img,nrows,ncols)
        
        observations, infos = self._make_info()
            
        self._make_jobs()
        return observations, infos

    def _make_info(self):
        observations = {}#agent: NONE for agent in self.agents}
        infos = {}#agent: {} for agent in self.agents}

        for idx,agv in enumerate(self.world.all_cranes):
            if SHARE.OBSERVATION_IMAGE:
                observations[agv.cfg.name]=self.world.get_observation_img(agv)
            else:
                observations[agv.cfg.name]=self.world.get_observation(agv)
            mask=self.world.get_masks(agv)
            
            infos[agv.cfg.name]={"action_masks":mask}
        self.observations = observations
        self.infos = infos
        # for name,obs,mask in zip(infos.keys(),observations.values(),infos.values()):
        #     self.observations[name]={'observation':obs,'action_masks':mask}
        return observations,infos

    def step(self, actions:dict):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        # if not actions:
        #     self.agents = []
        #     return {}, {}, {}, {}, {}
        # acts=[0]*len(actions)
        # for k,v in actions.items():
        #     idx=self.agent_name_mapping[k]
        #     acts[idx]=v

        acts=list(actions.values())

        
        self.world.set_commands(acts)

        self.world.update()
        nrows=self.renderer.nrows
        ncols=self.renderer.ncols
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            screen_img=self.render()
            self.world.get_state_img(screen_img,nrows,ncols)

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = self.world.rewards
        

        terminations = {agent: self.world.is_over for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        observations, infos = self._make_info()

        # if self.world.is_over:
        #     self.agents = []

        return observations, rewards, terminations, truncations, infos