from .random_select import RandomSelect
from epsim.core import CraneAction,SHARE
import numpy as np
import random
class MaskSelect(RandomSelect):

    def decision(self,infos:dict):
        actions={}
        for k,info in infos.items():
            #print(k,info)
            masks=info['action_masks']

            
            acts=np.argwhere(masks).ravel()
            actions[k]=random.choice(acts)
        return actions

