import numpy as np
import gymnasium
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from epsim.core.simulator import Simulator
from epsim.core import *
from epsim.monitors import JobHistory

def env(**kwargs):
    env = raw_env(**kwargs)
    # if env.continuous:
    #     env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "ep_sim_v1",
        "is_parallelizable": True,
        "render_fps": FPS,
        "has_manual_policy": True,
    }

    def __init__(self, render_mode,args):
        #EzPickle.__init__(self, **kwargs)

        self.game=Simulator(args,False)

        self.render_mode = render_mode
        self.renderer = None
        self.renderOn = False
        self.job_recorder=None

        self.agents = ['dispatch']
        # self.agents.extend(
        #     [info.code for info in args['CRANES']])

        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict( \
            zip( \
                self.agents, list(range(len(self.agents))) \
            ) \
        )
        self.state_space =None
        self._agent_selector = agent_selector(self.agents)

        
        

    def observation_space(self, agent):
        #if agent=='dispatch':
        return gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(MAX_NUM_MACHINE, 12, 7),
            dtype=np.float32
        )


    
    def observe(self, agent):
        rt=[]
        for crane_id, crane in self.world.get_component(Crane):
            rt.append(self.game.get_states(crane_id))

        # if len(rt)<1:
        #     return np.zeros((len(self.game.CRANES),12,7),dtype=np.float)
        return np.array(rt,dtype=np.float)

    def state(self):
        """Returns an observation of the global environment."""
        # state = pygame.surfarray.pixels3d(self.screen).copy()
        # state = np.rot90(state, k=3)
        # state = np.fliplr(state)
        return self.observe(None)
         

    def action_space(self, agent):
        #if agent=='dispatch':
        return gymnasium.spaces.Discrete(len(self.game.PROCS)+1)
        #return gymnasium.spaces.Discrete(21, start=0)



    def seed(self, seed=None):
        self.game.seed(seed)
        #self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed)
        self.game.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.observations = {}
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        
        self.terminations = {a: False for a in self.agents}
        self.truncations =  {a: False for a in self.agents}
        self.infos =  {a: {} for a in self.agents}
        if self.job_recorder is None:
            self.job_recorder=JobHistory(self.game)
        else:
            self.job_recorder.reset('',0)
        if self.render_mode == "human":
            self.render()



        
    def render(self):
        if self.renderer is None:
            from epsim.renderers.renderer import Renderer
            self.renderer = Renderer(self.game,self.job_recorder)
        rt = self.renderer.render(self.render_mode)
        if self.render_mode == "human":
            self.renderer.clock.tick(self.metadata["render_fps"])
        return rt

    def step(self, action):
        if (self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        # if not self.action_spaces[agent].contains(action):
        #     raise Exception(
        #         "Action for agent {} must be in Discrete({})."
        #         "It is currently {}".format(agent, self.action_spaces[agent].n, action)
        #     )

        self.game.step(action, agent)
        if self.game.is_fail :
            self.truncations[self.agent_selection]=True
        if self.game.is_done :
            self.terminations[self.agent_selection]=True    
        # select next agent and observe
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()
