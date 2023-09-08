import functools

import gymnasium
from gymnasium.spaces import Discrete,Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from ..render.renderer import Renderer
from epsim.core import World,WorldObj,Slot,Crane,Actions,SHARE
from epsim.core.componets import Color
from epsim.core import SHARE

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(render_mode=render_mode)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes":  ["human", "rgb_array"],"name": "electroplating_v1"}

    def __init__(self, render_mode=None,args: "DictConfig"  = None):
        self.args=args
        self.args=args
        self.world=World(args.config_directory,args.max_x)
        Renderer.LANG=args.language
        SHARE.TILE_SIZE=args.tile_size
        SHARE.CHECK_TIME1=args.alarm.warning
        SHARE.CHECK_TIME2=args.alarm.fatal

        ncols=args.screen_columns
        
        rows=int(args.max_x/ncols+0.5)+1
        self.renderer=Renderer(self.world,args.fps,rows,ncols)
        
        self.possible_agents = [crane.cfg.name for crane in self.world.all_cranes]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.world.all_cranes))))
        )
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(-1,1,((2*SHARE.MAX_AGENT_SEE_DISTANCE+1)*19,))

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        return self.renderer.render(self.render_mode)

    def close(self):
        self.renderer.close()


    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        
        self.world.reset()
        self.agents = self.possible_agents[:]
        
        observations = {}#agent: NONE for agent in self.agents}
        infos = {}#agent: {} for agent in self.agents}
        for idx,agv in enumerate(self.world.all_cranes):
            observations[agv.cfg.name]=self.world.get_observation(idx)
            infos[agv.cfg.name]={"action_masks":self.world.get_masks(agv)}
        self.state = observations
        ps=[]
        for p in self.args.products:
            ps.extend([p.code]*p.num)
        self.world.add_jobs(ps)
        for agv in self.world.all_cranes:
            agv.color=Color(255,255,255)
        self.world.cur_crane.color=Color(255,0,0)
        return observations, infos

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
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        acts=[0]*len(actions)
        for k,v in actions.items():
            idx=self.agent_name_mapping[k]
            acts[idx]=v

        
        self.world.set_commands(acts)

        self.world.update()

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {agent: self.world.reward for agent in self.agents}
        

        terminations = {agent: self.world.is_over for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        observations = {}
        infos = {}

        for idx,agv in enumerate(self.world.all_cranes):
            observations[agv.cfg.name]=self.world.get_observation(idx)
            infos[agv.cfg.name]=self.world.get_masks(agv)
        self.state = observations


        if self.world.is_over:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos