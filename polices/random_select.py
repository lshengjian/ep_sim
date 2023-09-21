from gymnasium import Env
import numpy as np
import random
from epsim.core import DispatchAction
class RandomSelect:
    def __init__(self,env:Env):
        self.env=env

    def decision(self,infos=None):
        actions={}
        for k,info in infos.items():
            actions[k]=self.env.action_space(k).sample()
        return actions

