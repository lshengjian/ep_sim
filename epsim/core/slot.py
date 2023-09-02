from .world_object import WorldObj
from .constants import *
from .componets import *
from .workpiece import Workpiece
from .shapes import get_slot_shape,get_progress_bar
from .rendering import set_color,blend_imgs
from .workpiece import Workpiece
class Slot(WorldObj):#缓存及加工位
    def __init__(self, x:int,cfg:SlotData ):#,ops_dict: Dict[int,OperateData]
        self.cfg:SlotData=cfg
        self.timer:int=0
        self.locked=False
        super().__init__(x)
        
    def __str__(self):
        #locked='X' if self.locked else ''
        old=super().__str__()
        return f'{self.cfg.op_name} {old})'
    
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
        self.carrying=None
        self.locked=False
        if wp is None or self.cfg.op_key<10:
            return wp,0
        
        
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
            img=blend_imgs(wp.image,img,(0,self.TILE_SIZE//2))
            if self.cfg.op_key>9:
                op_time=(wp.target_op_limit.min_time+wp.target_op_limit.max_time)//2
                left=op_time-self.timer
                p=int(self.timer/op_time*100+0.5)
                pg_bar=get_progress_bar(p)
                color=(0,255,0)
                if left<10:#max(op_time*0.1,5)
                    color=(255,0,0)
                elif left<20: #max(op_time*0.2,10)
                    color=(255,255,0)
                pg_bar=set_color(pg_bar,*color)
                img=blend_imgs(pg_bar,img,(int(self.TILE_SIZE*0.06),self.TILE_SIZE*2-pg_bar.shape[0]))

        return img    