import pygame

class ManualJobPolicy:
    def __init__(self, env):

        self.env = env

        self.default_action = 0
        self.select_action = -1

    def __call__(self,obs,info):

        action = self.default_action
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exit()

                elif event.key == pygame.K_BACKSPACE:
                    self.env.reset()

                elif event.key==pygame.K_TAB:
                    action=1

                elif event.key==pygame.K_SPACE:
                    
                    action=2
                    

        return action





