import numpy as np
from typing import Optional, Union,  Dict
import gym
from gym.spaces import *
from epsim.core.simulator import Simulator
from epsim.core import *
# from epsim.monitors import JobHistory

class MyEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]): 
    """
        State for game:

        Action for an agent:
        -5~-1:BACK n step
        0: NOOP
        1~5:FORWARD n step
   
    """
    metadata = {
        "render_modes": ["ansi","human", "rgb_array"],
        "render_fps":8,
    }
    def set_space(self):
        self.action_space = Discrete(1+len(self.game.PROCS))
        # nb_cranes=len(self.game.CRANES)
        # nb_jobs=len(self.game.jobs)
        # nb_pos=self.game.END+1 # 0 for wait plan
        # self.observation_space = \
        # Dict(
        #     {f"{i+1}": Box(low=0,high=255,shape=(VIEW_DISTANCE,9),dtype=np.uint8) for i in range(len(self.game.CRANES))}
        # )
    def __init__(self, render_mode,args):

        # self.screen_height, self.screen_width=CELL_SIZE*ROWS, \
        #     CELL_SIZE*(ROWS+INFO_COLS)

        self.game=Simulator(args)
        self.set_space()

        self.render_mode = render_mode
        self.metadata['render_fps']=FPS
        self.renderer = None
        self.renderOn = False
        self.state=None


    def action_mask(self, state):
        """Computes an action mask for the action space using the state information."""
        return self.game.get_mask(state)


    def make_state(self):
        data=self.game.get_state()
        self.state= np.array(data,dtype=float)

    

        
    def step(self, action):
        self.game.step(action)
        self.make_state()
        
        if self.render_mode == "human":
            self._render_gui()

        return self.state, \
            self.game.reward, self.game.is_done,self.game.is_fail, \
            {
                'score':self.game.total_reward,
                'action_mask':self.action_mask(self.state)}        
        
       

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        
        super().reset(seed=seed)
        self.game.reset(seed=seed)

        self.set_space()
        self.make_state()
       

        if self.render_mode == "human":
            self._render_gui()
        return self.state,{ 'score':0,
        'action_mask':self.action_mask(self.state)}
                
     



    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def render(self):

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()

    def _render_text(self):
        pass
    
    def _render_gui(self):
        if self.renderer is None:
            from epsim.renderers.renderer import Renderer
            self.renderer = Renderer(self.game)
        rt = self.renderer.render(self.render_mode)
        if self.render_mode == "human":
            self.renderer.clock.tick(self.metadata["render_fps"])
        return rt   