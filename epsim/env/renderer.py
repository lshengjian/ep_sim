import os,numpy as np
from gymnasium.error import DependencyNotInstalled
from ..core.world import World
from ..core.shapes import get_slot_shape ,get_crane_shape,get_workpiece_shape
from ..core.rendering import set_color,blend_imgs
from copy import deepcopy
RENDER_DIR=os.path.abspath(os.path.dirname(__file__) )

def get_font(pygame,font_size=16,shadow=False):
    if shadow:
        return pygame.font.Font(RENDER_DIR+'/fonts/WHARMBY.TTF', font_size)
    return pygame.SysFont('arial', font_size)


def make_surface(img,pygame):
    #img= np.transpose(img, axes=(1, 0, 2))
    img=img.swapaxes(0,1)
    return pygame.surfarray.make_surface(img)


class Renderer:
    def __init__(self,world:World,fps:int=4,nrow=3,ncol=17,tile_size=32):
        self.world:World=world
        self.fps=fps
        self._surface = None
        self.clock = None
        self.nrow=nrow
        self.ncol=ncol
        self.tile_size = tile_size
        self.window_size =  ncol*tile_size, tile_size*nrow*3

    
    def show_text(self,pygame,msg,x,y,fsize=16,shadow=False,color=(255, 221, 85)):
        font=get_font(pygame,fsize,shadow)
        textSerface = font.render(msg, True,color )#,(0,0,0)
        width, height = font.size(msg)
        self._surface.blit(textSerface, (x, y))
    
    def _draw_products(self,pygame):
        x=self.window_size[0]//2
        for code in self.world.products:
            img=get_workpiece_shape(code)
            if self.world.cur_prd_code==code:
                img=set_color(img,245,16,40)
            self._surface.blit(make_surface(img,pygame),(x,self.tile_size//2))
            x+=self.tile_size
        
    def _draw(self,pygame):
        #self._check_cache()
        #xs=list(map(lambda c:int(c.x+0.5) , self.world.all_cranes))
        self._surface.fill((0,0,0))
        self.show_text(pygame,f'R:{self.world.reward} S:{self.world.score} T:{self.world.step_count}',10,6,self.tile_size,True,(155,34,237))
        self._draw_products(pygame)
        merges=[]
        crane_offsets={}#:Dict[int,Crane]
        for crane in self.world.all_cranes:
            cx=int(crane.x+0.5)
            if  cx in self.world.pos_slots and crane.y>=1:
                merges.append(cx)
                crane_offsets[cx]=crane
        for x,s in self.world.pos_slots.items():
            r=x//self.ncol
            crane=x%self.ncol
            if x not in merges:
                self._surface.blit(make_surface(s.image,pygame),(crane*self.tile_size,self.tile_size*(3*r+3)))
            else:
                img2=deepcopy(s.image)
                img2[:,:,:]=s.image
                crane_img=crane_offsets[x].image
                img2=blend_imgs(crane_img,img2,(0,(crane_offsets[x].y-1)*self.tile_size))
                self._surface.blit(make_surface(img2,pygame),(crane*self.tile_size,self.tile_size*(3*r+3)))
            
                
        for crane in self.world.all_cranes:
            x=int(crane.x+0.5)
            if  x in merges: continue
            r=x//self.ncol
            c=x%self.ncol
            img=make_surface(crane.image,pygame)
            self._surface.blit(img,(c*self.tile_size,(crane.y+3*r+2)*self.tile_size))
           


    def render(self, mode:str):
        try:
            import pygame
            #from pygame import gfxdraw
            
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
            if self.fps>0:
                self.clock.tick(self.fps)
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
            )

    def close(self):
        pass

 