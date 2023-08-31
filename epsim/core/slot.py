from .world_object import WorldObj
from .constants import *
from .componets import *
from .workpiece import Workpiece
from .shapes import get_slot_shape
from .rendering import set_color,blend_imgs
from .workpiece import Workpiece
class Slot(WorldObj):#缓存及加工位
    def __init__(self, x:int,cfg:SlotData ):#,ops_dict: Dict[int,OperateData]
        self.cfg:SlotData=cfg
        self.timer:int=0
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

    def put_in(self,wp:Workpiece):
        if wp is None:
            return
        self.timer=0
        wp.attached=self
        self.carrying=wp
        self.locked=True

    
    def take_out(self)->Tuple:
        wp:Workpiece=self.carrying
        if wp is None:
            return None,0
        self.carrying=None
        self.locked=False
        op:OpLimitData=wp.target_op_limit
        r=0
        if op.min_time-3<self.timer<op.min_time :
            r=-1
        elif self.timer<=op.min_time-3:
            r=-3
        if op.max_time<self.timer<op.max_time+3:
            r=-1
        elif self.timer>=op.max_time+3:
            r=-3
        return wp,r    


    def step(self):
        if self.carrying!=None and self.cfg.op_key>9:
            self.timer+=1

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