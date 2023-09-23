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
            # if k==SHARE.DISPATCH_CODE:
            #     if np.random.random()<0.95:
            #         masks[2]=0
            acts=np.argwhere(masks).ravel()
            actions[k]=random.choice(acts) if len(acts)>0 else 0
        return actions

