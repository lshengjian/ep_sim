import numpy as np
import pygame


class ManualPolicy:
    def __init__(self, env):

        self.env = env
        self.agent_id = 0# len(env.agents)-1
        self.agent = self.env.agents[self.agent_id]

        self.default_action = 0
        self.action_mapping = dict()
        self.action_mapping[pygame.K_w] = 2#1.0
        self.action_mapping[pygame.K_s] = 0#-1.0

    def __call__(self,obs):
        # only trigger when we are the correct agent
        # assert (
        #     agent == self.agent
        # ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."

        # set the default action
        action = self.default_action

        # if we get a key, override action using the dict
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # escape to end
                    exit()

                elif event.key == pygame.K_BACKSPACE:
                    # backspace to reset
                    self.env.reset()

                elif event.key in self.action_mapping:
                    action = self.action_mapping[event.key]
                    print(f'{self.agent}:{action}')
                elif event.key==pygame.K_a:
                    self.agent_id-=1
                    if self.agent_id<0:
                        self.agent_id=0
                    self.agent = self.env.agents[self.agent_id]
                    #print(f'cur agent:{agent}')
                elif event.key==pygame.K_d:
                    self.agent_id+=1
                    if self.agent_id>len(self.env.agents)-1:
                        self.agent_id=len(self.env.agents)-1
                    self.agent = self.env.agents[self.agent_id]
                    #print(f'cur agent:{agent}')

        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping



