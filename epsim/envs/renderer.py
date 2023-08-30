import numpy as np
from gymnasium.error import DependencyNotInstalled
from ..core.world import World
from .shapes import get_slot_shape ,get_crane_shape
from .rendering import set_color,blend_imgs
from copy import deepcopy
# VISITE_COLOR=(55, 55, 55)
# AGENT_COLOR=(249, 12, 3)
# TEXT_COLOR=(0, 0, 0)
def make_surface(img,pg):
    img= np.transpose(img, axes=(1, 0, 2))
    return pg.surfarray.make_surface(img)


class Renderer:
    def __init__(self,world:World,FPS:int=4,nrow=3,ncol=17,tile_size=32):
        self.world:World=world
        self.FPS=FPS
        self._surface = None
        self.clock = None
        self.nrow=nrow
        self.ncol=ncol
        self.tile_size = tile_size
        self.window_size =  ncol*tile_size, tile_size*nrow*3
        self.slot_img_cache={}
        self.crane_img_cache={}

    def _check_cache(self):
        self.crane_img_cache.clear()
        for c in self.world.all_cranes:
            dir=c.action
            img=get_crane_shape(dir)
            r,g,b=255,0,0
            img=set_color(img,r,g,b)
            x=int(c.x+0.5)
            y=int(c.y+0.5)
            self.crane_img_cache[x]=(y,img)
  
        if len(self.slot_img_cache)>0:
            return
        for x,s in self.world.pos_slots.items():
            img=get_slot_shape(s.cfg.op_key)
            r,g,b=self.world.ops_dict[s.cfg.op_key].color.rgb
            img=set_color(img,r,g,b)
            self.slot_img_cache[x]=(1,img)


    def _draw(self,pygame):
        self._check_cache()
        #xs=list(map(lambda c:int(c.x+0.5) , self.world.all_cranes))
        self._surface.fill((0,0,0))
        merges=[]
        for x,d in self.crane_img_cache.items():
            if x in self.slot_img_cache and d[0]>0:
                merges.append(x)
        for x,s in self.world.pos_slots.items():
            _,slot_img=self.slot_img_cache[x]
            r=x//self.ncol
            c=x%self.ncol
            if x not in merges:

                self._surface.blit(make_surface(slot_img,pygame),(c*self.tile_size,self.tile_size*(1+3*r)))
            else:
                img2=deepcopy(slot_img)
                img2[:,:,:]=slot_img
                #np.zeros((self.tile_size*3,self.tile_size,3),dtype=np.uint8)
                #img2[self.tile_size:,:,:]=slot_img
                y,crane_img=self.crane_img_cache[x]
                img2=blend_imgs(crane_img,img2,(0,(y-1)*self.tile_size))
                self._surface.blit(make_surface(img2,pygame),(c*self.tile_size,self.tile_size*(1+3*r)))
            
                
        for x,d in self.crane_img_cache.items():
            if x in merges: continue
            r=x//self.ncol
            c=x%self.ncol
            y,img=self.crane_img_cache[x]
            img=make_surface(img,pygame)
            self._surface.blit(img,(c*self.tile_size,(y+3*r)*self.tile_size))
           


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

    def close(self):
        pass

 