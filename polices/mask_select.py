from .random_select import RandomSelect
from epsim.core import CraneAction
import numpy as np
import random
class MaskSelect(RandomSelect):

    def decision_ma(self,obs:dict,infos:dict):
        actions={}
        for k,info in infos.items():
            #print(info)
            masks=info['action_masks']
            if sum(masks)>1:
                masks[CraneAction.stay]=0
            #print(k,masks.tolist())
            
            acts=np.argwhere(masks).ravel()
            actions[k]=random.choice(acts)
        return actions

