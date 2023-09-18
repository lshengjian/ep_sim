from .workpiece import Workpiece
from .crane import Crane
from .slot import Slot
from .componets import *
from .constants import *
from typing import  List,Dict
import epsim.core.world as W

import logging
logger = logging.getLogger(__name__)


class Dispatch:
    def __init__(self,world): 
        self.world:W.World=world
        #print('init Dispatch')
    
    def decision(self):
        eps=SHARE.EPS
        cranes_bound={}
        for agv in self.world.all_cranes:
            cranes_bound[agv.cfg.id]=self.world.get_crane_bound(agv)
            self.world.masks[agv.cfg.id]=np.zeros(5,dtype=np.uint8)
            self.world.masks[agv.cfg.id][Actions.stay]=1
            agv.force=0

        
        for agv in self.world.all_cranes:
            masks=self.world.masks[agv.cfg.id]
            if self.check_middle(agv):
                #print(f'{agv} check_middle {masks}')
                continue
            bound=cranes_bound[agv.cfg.id]
            if  agv.y<eps :
                self.check_up_move(agv,bound)
                #print(f'{agv} check_up_move {masks}')
                
            elif agv.y>self.world.max_y-eps :
                self.check_down_move(agv,bound)
                #print(f'{agv} check_down_move {masks}')
            
            # self.check_bound(agv,bound)
            # print(f'{agv} check_bound {masks}')

    
    def check_middle(self,agv:Crane):
        masks = self.world.masks[agv.cfg.id]
        if self._limit_only_up(agv,masks) or self._limit_only_down(agv,masks):
            masks[Actions.stay]=0
            return True
        return False
    
    def check_up_move(self,crane:Crane,bound:Tuple[int,int,Crane,Crane] ):
        eps=SHARE.EPS
        wp:Workpiece=crane.carrying
        slot=self.world.pos_slots.get(crane.x,None)
        if  slot !=  None :
            masks=self.world.masks[crane.cfg.id]
            wp2:Workpiece=None if slot is None else slot.carrying
            if wp!=None and  wp.target_op_limit.op_key == slot.cfg.op_key and wp2==None:
                masks[Actions.bottom]=1 
                masks[Actions.stay]=0
                return
               
        slots=self.get_focus_slots(crane,bound)
        self.go_home_or_target(crane,  slots)

    def go_home_or_target(self, crane,  slots):
        masks=self.world.masks[crane.cfg.id]
        if len(slots)<1:
            if crane.init_x<crane.x:
                masks[Actions.left]=1
            elif crane.init_x>crane.x:
                masks[Actions.right]=1
        else :
            self.go_target(crane, slots)

    def go_target(self, crane:Crane, slots:List[Slot]):
        masks=self.world.masks[crane.cfg.id]
        #dir=set()
        for s in slots:
            dis=s.x-crane.x
            dir=dis/abs(dis)
            crane.force+=dir/(dis**2)
        if crane.force>0:
            masks[Actions.right]=1
        elif crane.force<0:
            masks[Actions.left]=1

            # if s.x<crane.x:
            #     dir.add(Actions.left)
            # elif s.x>crane.x:
            #     dir.add(Actions.right)
            
        # if Actions.left in dir : 
        #     masks[Actions.left]=1
        # if Actions.right in dir:
        #     masks[Actions.right]=1



    def check_down_move(self,crane:Crane,bound:Tuple[int,int,Crane,Crane] ):
        eps=SHARE.EPS
        wp:Workpiece=crane.carrying
        slot=self.world.pos_slots.get(crane.x,None)
        masks=self.world.masks[crane.cfg.id]
        wp2:Workpiece=None if slot is None else slot.carrying
        if wp2!=None:
            if slot.timer>=wp2.target_op_limit.duration  :
                x1,x2,left,right=bound
                next_slot=self.next_slot( wp2,x1,x2)
                if next_slot != None and (slot.cfg.op_key>SHARE.MIN_OP_KEY or np.random.random()<0.2):
                    masks[Actions.stay]=0
                    masks[Actions.top]=1
                    return


        slots=self.get_focus_slots(crane,bound)
        self.go_home_or_target(crane,  slots)

    

        
    def check_bound(self, crane:Crane,bound:Tuple[int,int,Crane,Crane] ):
        x1,x2,left,right=bound
        eps=SHARE.EPS
        masks=self.world.masks[crane.cfg.id]
        if crane.y<eps:
            masks[Actions.top]=0
        elif crane.y>(self.world.max_y-eps):
            masks[Actions.bottom]=0
        if eps<crane.y<(self.world.max_y-eps):
            masks[Actions.left]=0
            masks[Actions.right]=0
        
        if crane.x<=x1:
            masks[Actions.left]=0
        if crane.x>=x2:
            masks[Actions.right]=0
        
        
        slot=self.world.pos_slots.get(crane.x,None)
        if slot is None : #没有槽位
            masks[Actions.bottom]=0
            masks[Actions.top]=0
        elif slot.carrying is None:
            masks[Actions.top]=0


    def next_slot(self,  wp:Workpiece,x1, x2):
        found=None
        op_limit:OpLimitData=self.world._get_next_op_limit(wp)
        # if x1==x2:
        #     slot:Slot=wp.carrying
        #     x1,x2=self.world.group_limits(slot.cfg.group)
        
        for x in range(x1,x2+1):
            s=self.world.pos_slots.get(x,None)
            #print(s,'' if s==None else s.carrying)
            if s!=None and s.carrying is None and s.cfg.op_key==op_limit.op_key:
                found=s

                break
        return found

    def get_focus_slots(self,crane:Crane,bound:Tuple[int,int,Crane,Crane]):
        x1,x2,*_=bound
        cs1=[]
        cs2=[]

        for slot in self.world.group_slots[crane.cfg.group]:
            if slot.x<x1 or slot.x>x2 or slot.x == crane.x :continue
            wp2:Workpiece=slot.carrying
            wp:Workpiece=crane.carrying
            dis=abs(slot.x-crane.x)
            if crane.y<SHARE.EPS and wp !=None: #天车载物，
                if wp2 is None and wp.target_op_limit.op_key==slot.cfg.op_key: # 加工槽空闲且是目标位置
                    cs1.append((dis,slot))
            elif crane.y>(self.world.max_y-SHARE.EPS) and wp is None and wp2 != None:
               if slot.cfg.op_key>SHARE.MIN_OP_KEY and (slot.timer+dis/crane.cfg.speed_x>=wp2.target_op_limit.min_time):
                    next_slot=self.next_slot( wp2,x1, x2)
                    if next_slot!=None:
                        cs1.append((dis,slot))
               else:
                    next_slot=self.next_slot( wp2,x1, x2)
                    if next_slot!=None:
                        cs2.append((dis,slot))
        cs1.sort(key=lambda d:d[0])
        cs2.sort(key=lambda d:d[0])
        cs1.extend(cs2)
        if len(cs1)<1:
            return []
        return list(map(lambda d:d[1] ,cs1))
        
  

    def _limit_only_up(self,crane:Crane,masks:np.ndarray)->bool:
        rt=False
        if 0<crane.y<SHARE.MAX_Y and crane.carrying!=None:
            masks[Actions.top]=1
            rt=True
        return rt

    def _limit_only_down(self,crane:Crane,masks:np.ndarray)->bool:
        rt=False
        if 0<crane.y<SHARE.MAX_Y and crane.carrying is None:
            masks[Actions.bottom]=1
            rt=True
        return rt
