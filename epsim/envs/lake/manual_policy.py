import numpy as np
import pygame
'''
    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP
'''
class ManualPolicy:
    def __init__(self, env):

        self.env = env

        self.default_action = 2
        self.action_mapping = dict()
        self.action_mapping[pygame.K_LEFT] = 0
        self.action_mapping[pygame.K_DOWN] = 1
        self.action_mapping[pygame.K_RIGHT] = 2
        self.action_mapping[pygame.K_UP] = 3

    def __call__(self,obs):

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
                    print(f'action:{action}')


        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping



