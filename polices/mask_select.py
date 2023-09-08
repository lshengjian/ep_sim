from .random_select import RandomSelect
from epsim.core import Actions
import numpy as np
import random
class MaskSelect(RandomSelect):

    def decision(self,obs:dict,infos:dict):
        actions={}
        for k,info in infos.items():
            print(k,info)
            masks=info['action_masks']
            acts=np.argwhere(masks).reshape(-1)
            actions[k]=random.choice(acts)
        return actions
