from .world_object import WorldObj
from .constants import *
from .componets import *
from .workpiece import Workpiece
from .shapes import get_slot_shape
from .rendering import set_color,blend_imgs
class Slot(WorldObj):#缓存及加工位
    def __init__(self, x:int,cfg:SlotData ):#,ops_dict: Dict[int,OperateData]
        self.cfg:SlotData=cfg
        self.timer:int=0
        self.left_time:int=9999
        self.locked=False
        super().__init__(x)
        
    def __str__(self):
        locked='X' if self.locked else ''
        old=super().__str__()
        return f'{locked} {self.cfg.op_name} {old})'
    
    def reset(self):
        super().reset()
        self.timer=0
        self.locked=False

    def step(self):
        self.left_time=9999
        if self.carrying!=None and self.cfg.op_key>9:
            op:OpLimitData=self.carrying.target_op_limit
            self.timer+=1
            self.left_time=op.min_time-self.timer

    @property
    def image(self):
        img=get_slot_shape(self.cfg.op_key)
        r,g,b=self.color.rgb
        img=set_color(img,r,g,b)
        if self.carrying!=None:
            wp:Workpiece=self.carrying
            img_wp=wp.image
            img=blend_imgs(img_wp,img,(0,self.TILE_SIZE//2))

        return img    