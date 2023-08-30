import os,numpy as np
from gymnasium.error import DependencyNotInstalled
from ..core.world import World
from .shapes import get_slot_shape ,get_crane_shape,get_workpiece_shape
from .rendering import set_color,blend_imgs
from copy import deepcopy
RENDER_DIR=os.path.abspath(os.path.dirname(__file__) )

def get_font(pygame,font_size=16,shadow=False):
    if shadow:
        return pygame.font.Font(RENDER_DIR+'/fonts/WHARMBY.TTF', font_size)
    return pygame.SysFont('arial', font_size)


def make_surface(img,pygame):
    img= np.transpose(img, axes=(1, 0, 2))
    return pygame.surfarray.make_surface(img)


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
    
    def show_text(self,pygame,msg,x,y,fsize=16,shadow=False,color=(255, 221, 85)):
        font=get_font(pygame,fsize,shadow)
        textSerface = font.render(msg, True,color )#,(0,0,0)
        #pygame.image.save(text_serface, "text.png")
        # 绘制前先获取文本的宽高
        width, height = font.size(msg)
        # 绘制到显示器的surface上
        self._surface.blit(textSerface, (x, y))
        return width, height
    
    def _draw_products(self,pygame):
        x=self.window_size[0]//2
        for code,d in self.world.products.items():
            img=get_workpiece_shape(code)
            if self.world.cur_prd_code==code:
                img=set_color(img,245,216,40)
            for i in range(d[0]-d[1]):
                self._surface.blit(make_surface(img,pygame),(x,self.tile_size//2))
                x+=self.tile_size
        
    def _draw(self,pygame):
        self._check_cache()
        #xs=list(map(lambda c:int(c.x+0.5) , self.world.all_cranes))
        self._surface.fill((0,0,0))
        self.show_text(pygame,f'R:{self.world.reward:>02.0f} S:{self.world.score:>04.0f}',10,6,36,True,(155,34,237))
        self._draw_products(pygame)
        merges=[]
        for x,d in self.crane_img_cache.items():
            if x in self.slot_img_cache and d[0]>0:
                merges.append(x)
        for x,s in self.world.pos_slots.items():
            _,slot_img=self.slot_img_cache[x]
            r=x//self.ncol
            c=x%self.ncol
            if x not in merges:
                if s.carrying!=None:
                    wp_img=get_workpiece_shape(s.carrying.prouct_code)
                    slot_img=blend_imgs(wp_img,slot_img,(0,self.tile_size//2))
                self._surface.blit(make_surface(slot_img,pygame),(c*self.tile_size,self.tile_size*(3*r+3)))
            else:
                img2=deepcopy(slot_img)
                img2[:,:,:]=slot_img
                #np.zeros((self.tile_size*3,self.tile_size,3),dtype=np.uint8)
                #img2[self.tile_size:,:,:]=slot_img
                y,crane_img=self.crane_img_cache[x]
                img2=blend_imgs(crane_img,img2,(0,(y-1)*self.tile_size))
                self._surface.blit(make_surface(img2,pygame),(c*self.tile_size,self.tile_size*(3*r+3)))
            
                
        for x,d in self.crane_img_cache.items():
            if x in merges: continue
            r=x//self.ncol
            c=x%self.ncol
            y,img=self.crane_img_cache[x]
            img=make_surface(img,pygame)
            self._surface.blit(img,(c*self.tile_size,(y+3*r+2)*self.tile_size))
           


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
            if self.FPS>0:
                self.clock.tick(self.FPS)
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
            )

    def close(self):
        pass

 