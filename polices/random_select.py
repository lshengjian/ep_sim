from gymnasium import Env
import numpy as np
import random
from epsim.core import DispatchAction
class RandomSelect:
    def __init__(self,env:Env):
        self.env=env

    def decision_ma(self,obs,infos=None):
        actions={}
        for k,info in obs.items():
            actions[k]=self.env.action_space(k).sample()
        return actions

    def decision(self,obs:np.ndarray,info):
        acts=[DispatchAction.NOOP,DispatchAction.NEXT_PRODUCT_TYPE,DispatchAction.SELECT_CUR_PRODUCT]
        return random.choice(acts)