from .world_object import WorldObj
from .constants import *
from .componets import *
from .workpiece import Workpiece
from .shapes import get_slot_shape,get_progress_bar
from .rendering import set_color,blend_imgs
from .workpiece import Workpiece

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class Slot(WorldObj):#缓存及加工位

    def __init__(self, x:int,cfg:SlotData ):#,ops_dict: Dict[int,OperateData]
        assert x in cfg.offsets
        self.cfg:SlotData=cfg
        self.timer:int=0
        self.locked=False
        super().__init__(x)

    @property
    def state(self)->State:
        wp:Workpiece=self.carrying
        data=State(ObjType.Slot,self.cfg.op_key)
        data.x=self.x
        data.y=self.y
        if wp!=None:
            data=wp.state.clone()
        data.op_time=self.timer
        return data
            
    def __str__(self):
        #locked='X' if self.locked else ''
        old=super().__str__()
        return f'{old} {self.cfg.op_name}'
    
    def reset(self):
        super().reset()
        self.timer=0
        self.locked=False

    def put_in(self,wp:Workpiece):
        if wp is None:
            return
        self.timer=0
        #print(f'put {wp} to {self}')
        logger.debug(f'put {wp} to {self}')
        wp.attached=self
        self.carrying=wp
        self.locked=True
        

    
    def take_out(self)->Tuple:
        wp:Workpiece=self.carrying
        self.carrying=None
        self.locked=False
        if wp is None or self.cfg.op_key<SHARE.MIN_OP_KEY:
            return wp,0
        op:OpLimitData=wp.target_op_limit
        op_time=(wp.target_op_limit.min_time+wp.target_op_limit.max_time)//2
        if abs(self.timer-op_time)<3:
            return wp,2
        elif op.min_time<=self.timer<op.max_time:
            return wp,1
        r=0
        if op.min_time-SHARE.SHORT_ALARM_TIME<self.timer<op.min_time :
            r=-1
        elif self.timer<=op.min_time-SHARE.SHORT_ALARM_TIME:
            r=-3
        if op.max_time<self.timer<op.max_time+SHARE.LONG_ALARM_TIME:
            r=-2
        elif self.timer>=op.max_time+SHARE.LONG_ALARM_TIME:
            r=-5
        #print(f'tack out {wp}  from {self}')
        return wp,r    


    def step(self):
        self.timer+=1
        if self.carrying is None or  self.cfg.op_key<SHARE.MIN_OP_KEY:
            return
        wp:Workpiece=self.carrying
        wp.total_op_time+=1
        
        # op:OpLimitData=wp.target_op_limit
        # op_time=(wp.target_op_limit.min_time+wp.target_op_limit.max_time)//2
        # self.left=op_time-self.timer



    @property
    def image(self):
        img=get_slot_shape(self.cfg.op_key)
        r,g,b=self.color.rgb
        img=set_color(img,r,g,b)
        if self.carrying!=None:
            wp:Workpiece=self.carrying
            img=blend_imgs(wp.image,img,(0,SHARE.TILE_SIZE//2))
            if self.cfg.op_key>=SHARE.MIN_OP_KEY:
                op_time=(wp.target_op_limit.min_time+wp.target_op_limit.max_time)//2
                #self.left=op_time-self.timer
                p=int(self.timer/op_time*100+0.5)
                pg_bar=get_progress_bar(p)
                color=(0,255,0)
                if self.timer+SHARE.SHORT_ALARM_TIME>=wp.target_op_limit.max_time:#max(op_time*0.1,5)
                    color=(255,0,0)
                elif self.timer+SHARE.LONG_ALARM_TIME>=wp.target_op_limit.max_time: #max(op_time*0.2,10)
                    color=(255,255,0)
                pg_bar=set_color(pg_bar,*color)
                img=blend_imgs(pg_bar,img,(int(SHARE.TILE_SIZE*0.06),SHARE.TILE_SIZE*2-pg_bar.shape[0]))

        return img    