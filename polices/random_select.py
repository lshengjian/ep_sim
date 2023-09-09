from gymnasium import Env
class RandomSelect:
    def __init__(self,env:Env):
        self.env=env

    def decision(self,obs,infos=None):
        actions={}
        for k,info in obs.items():
            actions[k]=self.env.action_space(k).sample()
        return actions

