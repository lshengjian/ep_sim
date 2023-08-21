import pygame
import esper
from epsim.core import EVENT_CLICKED
class ManualJobPolicy:
    def __init__(self, env):

        self.env = env
        self.job_mgr = env.unwrapped.game.job_mgr


        self.select_action = -1

    def __call__(self,obs):

        action = 0

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_presses = pygame.mouse.get_pressed()
                if mouse_presses[0]:
                    pos=pygame.mouse.get_pos()
                    esper.dispatch_event(EVENT_CLICKED,pos)
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