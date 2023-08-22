import numpy as np
from os import path
from gymnasium.error import DependencyNotInstalled
from .config import *
from .world import World
VISITE_COLOR=(55, 55, 55)
AGENT_COLOR=(249, 12, 3)
TEXT_COLOR=(0, 0, 0)
class Renderer:
    def __init__(self,world:World,FPS:int=4):
        self.world=world
        self.desc = self.world.desc.tolist()
        assert isinstance(self.desc, list), f"desc should be a list or an array, got {self.desc}"
        self.FPS=FPS
        self._surface = None
        self.clock = None
        nrow=self.world.nrow
        ncol=self.world.ncol
        # pygame utils
        self.window_size = WINDOWS_SIZE
        self.cell_size =  (WINDOWS_SIZE[0] // ncol, WINDOWS_SIZE[1] // nrow)

    def render(self, mode:str,visits:np.ndarray,V:np.ndarray=None):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self._surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("My Mini Grid")
                self.font  =  pygame.font.Font(None,26)
                self._surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self._surface = pygame.Surface(self.window_size)

        assert (
            self._surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        #self._surface.fill((0, 0, 0))
        desc = self.desc
        nrow=self.world.nrow
        ncol=self.world.ncol
        w,h=self.cell_size
        r=min(w,h)/2
        for y in range(nrow):
            for x in range(ncol):
                pos = [x * w, y * h]
                rect = (*pos, *self.cell_size)
                flag=desc[y][x]
                flag = flag.decode()
                color=COLORS[flag]
                gfxdraw.box(self._surface,rect,color)
                s=self.world.idx2state(y,x)
                total=sum(visits[s])
                if visits[s,STAY]:
                    gfxdraw.aacircle(
                        self._surface,
                        int(pos[0]+w/2),
                        int(pos[1]+h/2),
                        int(r / 2*visits[s,STAY]/total),
                        VISITE_COLOR
                    )
                for k,d in ACT_DIRS.items():
                    if visits[s,k]:
                        scale=visits[s,k]/total
                        x0,y0=pos[0]+w//2,pos[1]+h//2
                        dx,dy=d[0]*w//2,d[1]*h//2
                        dx*=scale
                        dy*=scale
                        gfxdraw.line(self._surface,x0,y0,x0+int(dx),y0+int(dy),VISITE_COLOR)
                
                suf=self.font.render(f'{V[s]:.2f}',1,VISITE_COLOR,color)
                pos[0]+=self.cell_size[0]*0.05
                pos[1]+=self.cell_size[1]*0.05
                self._surface.blit(suf,pos)
        
        # paint the elf
        bot_row, bot_col = self.world.state2idx(self.world.state)
        pos = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        #last_action = self.world.lastaction if self.world.lastaction is not None else 0
        gfxdraw.filled_circle(
            self._surface,
            int(pos[0]+w/2),
            int(pos[1]+h/2),
            int(r / 2),
            AGENT_COLOR
        )
        for i in range(1,nrow):
            yi=i*h
            gfxdraw.hline(self._surface,0,self.window_size[0],yi,TEXT_COLOR)
        for i in range(1,ncol):
            xi=i*w
            gfxdraw.vline(self._surface,xi,0,self.window_size[1],TEXT_COLOR)

        
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            if self.FPS>0:
                self.clock.tick(self.FPS)
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
            )



 