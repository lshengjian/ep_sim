import numpy as np
from gymnasium.error import DependencyNotInstalled
from ..core.world import World
from .shapes import get_slot_shape ,get_crane_shape
from .rendering import set_color,blend_imgs
# VISITE_COLOR=(55, 55, 55)
# AGENT_COLOR=(249, 12, 3)
# TEXT_COLOR=(0, 0, 0)
def make_surface(img,pg):
    img= np.transpose(img, axes=(1, 0, 2))
    return pg.surfarray.make_surface(img)


class Renderer:
    def __init__(self,world:World,FPS:int=4,nrow=3,ncol=32,tile_size=32):
        self.world:World=world
        self.FPS=FPS
        self._surface = None
        self.clock = None
        self.nrow=nrow
        self.ncol=ncol
        self.tile_size = tile_size
        self.window_size =  ncol*tile_size, tile_size*nrow*3

    def _draw(self,pygame):
        xs=list(map(lambda c:int(c.x+0.5) , self.world.all_cranes))
        imgs={}
        self._surface.fill((0,0,0))
        for x,s in self.world.pos_slots.items():
            img=get_slot_shape(s.cfg.op_key)
            r,g,b=self.world.ops_dict[s.cfg.op_key].color.rgb
            img=set_color(img,r,g,b)
            
            if x in xs:
                imgs[x]=np.zeros((self.tile_size*3,self.tile_size,3),dtype=np.uint8)
                imgs[x][self.tile_size:,:,:]=img
            else:
                img=make_surface(img,pygame)
                self._surface.blit(img,(x*self.tile_size,self.tile_size))
        for c in self.world.all_cranes:
            dir=c.action
            if dir>0:dir-=1
            img=get_crane_shape(dir)
            r,g,b=255,0,0
            img=set_color(img,r,g,b)
            x=int(c.x+0.5)
            y=int(c.y+0.5)
            if y==0 or x not in imgs:
                img=make_surface(img,pygame)
                self._surface.blit(img,(x*self.tile_size,y))
                
            elif x in imgs:
                img=blend_imgs(img,imgs[x],(0,(y-1)*self.tile_size))
                img=make_surface(img,pygame)
                self._surface.blit(img,(x*self.tile_size,0))
            


    def render(self, mode:str):
        try:
            import pygame
            
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self._surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("electroplate simulator")
                #self.font  =  pygame.font.Font(None,26)
                self._surface = pygame.display.set_mode(self.window_size ,pygame.DOUBLEBUF, 32)
            elif mode == "rgb_array":
                self._surface = pygame.Surface(self.window_size,pygame.SRCALPHA, 32)

        assert (
            self._surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._draw(pygame)
        

        
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            if self.FPS>0:
                self.clock.tick(self.FPS)
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
            )



 