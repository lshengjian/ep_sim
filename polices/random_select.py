from gymnasium import Env
class RandomSelect:
    def __init__(self,env:Env):
        self.env=env

    def decision(self,obs,info=None):
        return self.env.action_space.sample()

