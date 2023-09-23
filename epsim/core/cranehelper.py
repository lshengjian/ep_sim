from .workpiece import Workpiece
from .crane import Crane
from .slot import Slot
from .componets import *
from .constants import *
from typing import  List,Dict
import epsim.core.world as W

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class CraneHelper:
    def __init__(self,world): 
        self.world:W.World=world
        #print('init Dispatch')
    def decision(self):
        eps=SHARE.EPS
        cranes_bound={}
        self.reset(cranes_bound)
        self._check(cranes_bound)
        self._push()
       
        for crane in self.world.all_cranes:
            masks=self.world._masks[crane.cfg.name]
            if crane._lock!=None:
                continue
            r,l=crane.forces
            t=r+l
            if t<eps:
                continue
            if np.random.random()<=r/t:
                masks[CraneAction.right]=1 
            else:
                masks[CraneAction.left]=1    
            
            self.check_bound(crane,cranes_bound[crane.cfg.name])
            #logger.info(f'{crane} force:{crane.force:.1f} masks:{masks}')
    def _check(self,cranes_bound):
        eps=SHARE.EPS
        for crane in self.world.all_cranes:
            masks=self.world._masks[crane.cfg.name]
            if  self.check_middle(crane):
                continue
            if crane._lock!=None: continue

            bound=cranes_bound[crane.cfg.name]
            if  crane.y<eps :
                self.check_up_move(crane,bound)
                #print(f'{agv} check_up_move {masks}')
                
            elif crane.y>self.world.max_y-eps :
                self.check_down_move(crane,bound)
                #print(f'{agv} check_down_move {masks}')    
    
    def _push(self):
        for crane in self.world.all_cranes:
            if 0<crane.y<self.world.max_y:
                continue
            for agv in self.world.group_cranes[crane.cfg.group]:
                dis =  abs(crane.x-agv.x)
                if crane==agv or dis>SHARE.MIN_CRANE_SAFE_DISTANCE*2:continue
                
                f=10*(crane.x-agv.x)/dis**2
                crane.add_force(f)
                logger.info(f'{agv} add force:{f:.1f} to {crane}')



    def reset(self, cranes_bound):
        for crane in self.world.all_cranes:
            crane.resert_force()
            if crane._lock!=None: continue
            cranes_bound[crane.cfg.name]=self.world.get_crane_bound(crane)
            self.world._masks[crane.cfg.name]=np.zeros(5,dtype=np.uint8)
            self.world._masks[crane.cfg.name][CraneAction.stay]=1
            

    
    def check_middle(self,agv:Crane):
        masks = self.world._masks[agv.cfg.name]
        if self._limit_only_up(agv,masks) or self._limit_only_down(agv,masks):
            masks[CraneAction.stay]=0
            return True
        return False
    
    def check_up_move(self,crane:Crane,bound:Tuple[int,int,Crane,Crane] ):
        eps=SHARE.EPS
        wp:Workpiece=crane.carrying
        masks=self.world._masks[crane.cfg.name]
        slot=self.world.pos_slots.get(crane.x,None)
        #print(f'before  masks:{masks}')
        if  slot !=  None :
            wp2:Workpiece=None if slot is None else slot.carrying
            if wp!=None and  wp.target_op_limit.op_key == slot.cfg.op_key and wp2==None:
                masks[CraneAction.bottom]=1 
              
        slots=self.get_focus_slots(crane,bound)
        self.go_home_or_target(crane,  slots)
        #print(f'before  masks:{masks}')

    def go_home_or_target(self, crane,  slots):
        masks=self.world._masks[crane.cfg.name]
        if len(slots)<1:
            if crane.init_x<crane.x:
                masks[CraneAction.left]=1
            elif crane.init_x>crane.x:
                masks[CraneAction.right]=1
        else :
            self.go_target(crane, slots)

    def go_target(self, crane:Crane, slots:List[Slot]):
        masks=self.world._masks[crane.cfg.name]
        x1,x2=self.world.group_limits[crane.cfg.group]
        w=abs(x2-x1)
        cs=[]
        for s in slots:
            dis=s.x-crane.x
            dir=dis/abs(dis)
            k=1
            wp2:Workpiece=s.carrying
            if wp2!=None :
                if s.cfg.op_key>SHARE.MIN_OP_KEY:
                    k=10*(s.timer/wp2.target_op_limit.duration)**10
                else:
                    k=0.5
            f=k*dir/(1+abs(dis)/w)
            logger.info(f'{s} add force:{f:.1f} to {crane}')
            crane.add_force(f)
            cs.append((abs(f),s))
        cs.sort(key=lambda d:d[0],reverse=True)
        if len(cs)>0:
            slot=cs[0][1]
            crane.lock(slot)






    def check_down_move(self,crane:Crane,bound:Tuple[int,int,Crane,Crane] ):
        eps=SHARE.EPS
        wp:Workpiece=crane.carrying
        slot=self.world.pos_slots.get(crane.x,None)
        masks=self.world._masks[crane.cfg.name]
        wp2:Workpiece=None if slot is None else slot.carrying
        if wp2!=None and slot.timer>=wp2.target_op_limit.duration  :
            op_limit:OpLimitData=wp2.target_op_limit
            x1,x2,left,right=bound
            next_slot=self.next_slot( wp2,x1,x2)
            if next_slot != None :
                if slot.cfg.op_key>SHARE.MIN_OP_KEY and slot.timer>=op_limit.duration :
                    masks[CraneAction.stay]=0
                    masks[CraneAction.top]=1
                    return
                else:
                    masks[CraneAction.top]=1

        slots=self.get_focus_slots(crane,bound)
        self.go_home_or_target(crane,  slots)

    

        
    def check_bound(self, crane:Crane,bound:Tuple[int,int,Crane,Crane] ):
        x1,x2,left,right=bound
        eps=SHARE.EPS
        masks=self.world._masks[crane.cfg.name]
        if crane.y<eps:
            masks[CraneAction.top]=0
        elif crane.y>(self.world.max_y-eps):
            masks[CraneAction.bottom]=0
        if eps<crane.y<(self.world.max_y-eps):
            masks[CraneAction.left]=0
            masks[CraneAction.right]=0
        
        if crane.x<=x1:
            masks[CraneAction.left]=0
        if crane.x>=x2:
            masks[CraneAction.right]=0
        
        
        slot=self.world.pos_slots.get(crane.x,None)
        if slot is None : #没有槽位
            masks[CraneAction.bottom]=0
            masks[CraneAction.top]=0
        elif slot.carrying is None and crane.carrying is None :
            masks[CraneAction.top]=0


    def next_slot(self,  wp:Workpiece,x1, x2):
        found=None
        op_limit:OpLimitData=self.world._get_next_op_limit(wp)
        for x in range(x1,x2+1):
            s=self.world.pos_slots.get(x,None)
            #print(s,'' if s==None else s.carrying)
            if s!=None and s.carrying is None and s.cfg.op_key==op_limit.op_key:
                found=s
                break
        return found

    def get_focus_slots(self,crane:Crane,bound:Tuple[int,int,Crane,Crane]):
        x1,x2,*_=bound
        todos=[]
        cs=[]
        for slot in self.world.group_slots[crane.cfg.group]:
            if slot.x<x1 or slot.x>x2 or slot.x == crane.x :continue
            wp:Workpiece=crane.carrying
            wp2:Workpiece=slot.carrying
            if crane.y<SHARE.EPS and wp !=None: #天车载物，
                if wp2 is None and wp.target_op_limit.op_key==slot.cfg.op_key: # 加工槽空闲且是目标位置
                    #print(slot)
                    todos.append(slot)
            elif crane.y>(self.world.max_y-SHARE.EPS) and wp is None and wp2 != None:
               next_slot=self.next_slot( wp2,x1, x2)
               if next_slot is None: continue
               if slot.cfg.op_key>SHARE.MIN_OP_KEY and slot.timer>=wp2.target_op_limit.duration-10:
                   todos.append(slot)
               else: #if np.random.random()<0.05:
                   cs.append(slot)
        return todos+cs
        
  

    def _limit_only_up(self,crane:Crane,masks:np.ndarray)->bool:
        rt=False
        if 0<crane.y<SHARE.MAX_Y and crane.carrying!=None:
            masks[CraneAction.top]=1
            rt=True
        return rt

    def _limit_only_down(self,crane:Crane,masks:np.ndarray)->bool:
        rt=False
        if 0<crane.y<SHARE.MAX_Y and crane.carrying is None:
            masks[CraneAction.bottom]=1
            rt=True
        return rt
