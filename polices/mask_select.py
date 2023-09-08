from .random_select import RandomSelect
from epsim.core import Actions
import numpy as np
import random
class MaskSelect(RandomSelect):

    def decision(self,obs,info):
        masks=info['action_masks']
        acts=np.argwhere(masks).reshape(-1)
        return random.choice(acts)
