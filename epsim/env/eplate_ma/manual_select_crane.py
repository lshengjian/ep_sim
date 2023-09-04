import numpy as np
import pygame
import esper
from epsim.core import EVENT_CLICKED
class ManualCranePolicy:
    def __init__(self, env):

        self.env = env
        self.default_action = 0
        self.crane_move=(None,0)
        esper.set_handler('crane_move',self.do_crane_move)

    def do_crane_move(self,crane_move):
        self.crane_move=crane_move
    def __call__(self,obs):

        action = self.default_action
        self.agent=self.crane_move[0]
        action=self.crane_move[1]
        self.crane_move=(None,0)
        #print(self.agent,action)

        return action

    # @property
    # def available_agents(self):
    #     return self.env.agent_name_mapping



