from .random_select import RandomSelect
from epsim.core import Actions
import numpy as np
import random
class MaskSelect(RandomSelect):

    def decision(self,obs:dict,infos:dict):
        actions={}
        for k,info in infos.items():
            #print(info)
            masks=info['action_masks']
            #print(k,masks.tolist())
            
            acts=np.argwhere(masks).ravel()
            actions[k]=random.choice(acts)
        return actions
