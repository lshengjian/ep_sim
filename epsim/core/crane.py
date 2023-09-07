from .world_object import WorldObj
from .constants import *
from .shapes import get_crane_shape
from .rendering import set_color,blend_imgs
from .componets import CraneData,State
from .workpiece import Workpiece

class Crane(WorldObj):
    def __init__(self,  x:int,cfg:CraneData):
        self.cfg:CraneData=cfg
        self.timer:int=0
        self.action:Actions=Actions.stay
        self.last_action:Actions=Actions.stay
        super().__init__(x)
    
    @property
    def state(self)->State:
        wp:Workpiece=self.carrying
        data=State(ObjType.Crane)
        if wp!=None:
            data=wp.state.clone()
            data.obj_type=ObjType.Crane
        return data  
         
    def __str__(self):
        return super().__str__()+f' {self.cfg.name} {Directions[self.last_action]}'

    def reset(self):
        super().reset()
        self._y=2.0
        self.timer=0
        self.action=Actions.stay
        self.last_action=Actions.stay

    def set_command(self,act:Actions):
        self.action=act
        self.last_action=act

    def step(self):
        if self.action==Actions.stay:
            return
        dir=DIR_TO_VEC[self.action]
        self._x=self.x+dir[0]*self.cfg.speed_x
        self._y=self.y+dir[1]*self.cfg.speed_y
        # self._y=np.clip(self._y,0,2)
        self.action=Actions.stay

    def put_in(self,wp:Workpiece):
        if wp is None:
            return
        wp.attached=self
        self.carrying=wp

    
    def take_out(self)->tuple:
        rt=self.carrying
        self.carrying=None
        return rt,0     
      
    @property
    def image(self):
        img=get_crane_shape(self.last_action)
        r,g,b=self.color.rgb
        img=set_color(img,r,g,b)
        if self.carrying!=None:
            wp:Workpiece=self.carrying
            img_wp=wp.image
            img=blend_imgs(img_wp,img,(0,0))

        return img   
         
      