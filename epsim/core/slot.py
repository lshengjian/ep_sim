from .world_object import WorldObj
from .constants import *
from .componets import *

class Slot(WorldObj):#缓存及加工位
    def __init__(self, x:int,cfg:SlotData ):#,ops_dict: Dict[int,OperateData]
        self.cfg:SlotData=cfg
        self.timer:int=0
        self.left_time:int=9999
        self.locked=False
        # self.ops_dict=ops_dict
        # self.op_key=0
        # for key,v in ops_dict.items():
        #     if v.name==cfg.name:
        #        self.op_key=key
        #        break 
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
        if self.carrying!=None:
            op:ProcessData=self.carrying.target_op
            self.timer+=1
            self.left_time=op.time_min-self.timer

