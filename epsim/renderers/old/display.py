
import numpy as np
from epsim.core.componets import *
from epsim.core.consts import *

cranes=None
slots= None
hud=None  
def close(env):
    if env.window is not None:
        import pygame

        pygame.display.quit()
        pygame.quit()

def _init_display(env):
    row,col=GRID_SIZE
    env.screen_width = (col+8.5)*CELL_W
    env.screen_height = row*CELL_W+MSG_HEIGHT
    env.clock = None
    env.isopen = True


def setup(env,mode):
    global cranes,slots,hud
    import pygame
    pygame.init()
    pygame.key.set_repeat(1, 1)
    pygame.display.init()
    pygame.display.set_caption("My Gym Demo")    


    from epsim.renderers.cranes import CraneRenderder
    from epsim.renderers.old.hud import HUD
    from epsim.renderers.slots import SlotRenderder
    _init_display(env)

    if env.clock is None:
        env.clock = pygame.time.Clock()
    WINDOW_SIZE=(env.screen_width, env.screen_height)
    if mode == "human":
        env.window = pygame.display.set_mode(WINDOW_SIZE)
    elif mode == "rgb_array":
        env.window = pygame.Surface(WINDOW_SIZE)
    cranes=CraneRenderder(env.game,env.window)
    slots=SlotRenderder(env.game,env.window)
    hud=HUD(env.game,env.window)

    
    

def show(env):
    import pygame
    mode=env.render_mode
    #world=env.world

    env.window.fill((0, 0, 0))
    slots.render()
    cranes.render()
    hud.render(job_data=env.job_recorder.data)


    if mode == "human":
        pygame.event.pump()
        pygame.display.flip()
        #pygame.display.update()
        env.clock.tick(env.metadata["render_fps"])
    elif mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.window)), axes=(1, 0, 2)
        )