from .workpiece import Workpiece
from .crane import Crane
from .slot import Slot
from .componets import *
from .constants import *
import epsim.core.world as W

import logging
logger = logging.getLogger(__name__)


class Dispatch:
    def __init__(self,world): 
        self.world:W.World=world
        print('init Dispatch')
    
    def decision(self):
        cranes_bound={}
        for agv in self.world.all_cranes:
            cranes_bound[agv.cfg.id]=self.world.get_crane_bound(agv)
            self.world.masks[agv.cfg.id]=np.ones(5,dtype=np.uint8)



        for agv in self.world.all_cranes:
            if self.check_updown(agv):
                continue
            bound=cranes_bound[agv.cfg.id]
            #self.check_slots(agv,bound)
            self.check_safe(agv,bound)

            

    def check_safe(self, crane:Crane,bound:Tuple[int,int,Crane,Crane] ):
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
        
        wp:Workpiece=crane.carrying
        slot=self.world.pos_slots.get(crane.x,None)
        wp2:Workpiece=None if slot is None else slot.carrying
        
        if slot is None :
            masks[Actions.bottom]=0
            masks[Actions.top]=0
            return
        if  crane.y<eps :
            if wp!=None and ( wp.target_op_limit.op_key != slot.cfg.op_key or wp2!=None):
                masks[Actions.bottom]=0
                return
        if crane.y<self.world.max_y:
            return
        
        if wp2 is None :
            masks[Actions.top]=0
            return

        if not self.have_next_slot( wp2,x1, x2)  :
            masks[Actions.top]=0
        elif slot.timer<wp2.target_op_limit.min_time   :
            masks[Actions.top]=0

    def have_next_slot(self,  wp,x1, x2):
        found=False
        op_limit:OpLimitData=self.world._get_next_op_limit(wp)
        for x in range(x1,x2+1):
            s=self.world.pos_slots.get(x,None)
            if s!=None and s.carrying is None and s.cfg.op_key==op_limit.op_key:
                found=True
                break
        return found

            

        


    def check_updown(self,agv:Crane):
        masks = np.zeros(5, dtype=np.int8)
        if self._limit_only_up(agv,masks) or self._limit_only_down(agv,masks):
            self.world.masks[agv.cfg.id]=masks
            return True
        return False

    def get_slots(self,crane:Crane,isFree:bool):
        cs1=[]
        cs2=[]

        for slot in self.world.group_slots[crane.cfg.group]:
            wp2:Workpiece=slot.carrying
            wp:Workpiece=crane.carrying
            if isFree:
                if wp2 is None:
                    if  wp.target_op_limit.op_key==slot.cfg.op_key and crane.y<SHARE.EPS: #天车载物，加工槽空闲
                        cs1.append(slot)
                    
                        #print('天车载物,左边可放')
            else:
                if wp2 != None:
                    if slot.cfg.op_key>SHARE.MIN_OP_KEY and slot.timer>wp.target_op_limit.duration-10:
                        cs1.append(slot)
                    else:
                        cs2.append(slot)
        cs1.extend(cs2)
        return filter(lambda slot:crane.cfg.group==slot.cfg.group and slot.carrying==None if isFree else slot.carrying!=None ,self.world.all_cranes)
    
    def check_slots(self,crane:Crane,bound:Tuple[int,int,Crane,Crane] ):
        pass
        # free_agvs=filter(lambda agv:agv.carrying==None,self.world.all_cranes)
        # work_agvs=filter(lambda agv:agv.carrying!=None,self.world.all_cranes)

        # for agv in free_agvs:
        #     slots=self.get_slots(agv,True)

        
        # 
        # if self._should_up(crane,masks,left,right):
        #     self.masks[crane.cfg.name]=masks
        #     #self._not_out_bound(crane,masks,left,right)
        #     return masks
        # if self._should_down(crane,masks):
        #     self.masks[crane.cfg.name]=masks
        #     return masks
        # if self._should_move(crane,masks,left,right):
        #     self._not_out_bound(crane,masks,left,right)
        #     self.masks[crane.cfg.name]=masks
        #     if sum(masks)<1:
        #         masks[Actions.stay]=1
        #     return masks

        # masks = np.ones(5, dtype=np.int8)
        # masks[Actions.top]=0
        # masks[Actions.bottom]=0
        # self._not_out_bound(crane,masks,left,right)
        # self.masks[crane.cfg.name]=masks




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

    def _should_move(self, crane, masks,left,right):
        eps=SHARE.EPS
        wp:Workpiece=crane.carrying
        cs1=[]
        cs2=[]
       
        can_go=set()
        for x in range(left.x,right.x+1):
            slot=self.pos_slots.get(x,None)
            if slot is None:continue
            wp2:Workpiece=slot.carrying
            if wp != None:
                if wp2 is None  and wp.target_op_limit.op_key==slot.cfg.op_key and crane.y<eps: #天车载物，加工槽空闲
                    if slot.x>crane.x:
                        can_go.add(Actions.right)
                        #print('天车载物,右边可放')
                        
                    elif slot.x<crane.x:
                        can_go.add(Actions.left)
                        #print('天车载物,左边可放')
            else:
                if wp2!=None and slot.timer>wp2.target_op_limit.duration-10:
                    if slot.cfg.op_key>SHARE.MIN_OP_KEY:
                        cs1.append(slot)
                    else:
                        cs2.append(slot)
        cs1.extend(cs2)
        for  slot in cs1:
            if slot.x>crane.x:
                can_go.add(Actions.right)
                #print('天车空,右边需要处理')
                if slot.carrying!=None :
                    if type(right) is Crane and right.carrying is None:
                        break
                else:
                    can_go.add(Actions.right)
            elif slot.x<crane.x:
                can_go.add(Actions.left)
                #print('天车空,左边需要处理')
                if slot.carrying!=None :
                    
                    if type(left) is Crane and left.carrying is None:
                        break
 


        rt=False
        if Actions.left in can_go :
            masks[Actions.left]=1
            rt=True
        if Actions.right in can_go:
            masks[Actions.right]=1
            rt=True
        return rt

          






    def _should_down(self, crane, masks):
        eps=SHARE.EPS
        wp=crane.carrying
        slot=self.pos_slots.get(crane.x,None)
        wp2=None if slot is None else slot.carrying
        if  crane.y<eps :
            candown=wp != None and slot!=None and wp2 ==None and wp.target_op_limit.op_key==slot.cfg.op_key 
            if  candown:
                masks[Actions.bottom]=1
                return True
        return False


    def _should_up(self, crane:Crane, masks,left,right):
        eps=SHARE.EPS
        wp=crane.carrying

        if crane.y<self.max_y or wp!=None:
            return

        slot=self.pos_slots.get(crane.x,None)
        wp2=None if slot is None else slot.carrying
        if wp!=None or wp2 is None: #空天车在空槽不能上行
            return

        found=False
        op_limit:OpLimitData=self._get_next_op_limit(wp2)
        for x in range(left.x,right.x+1):
           s=self.pos_slots.get(x,None)
           wp=None if s is None else s.carrying
           if s!=None and wp is None and s.cfg.op_key==op_limit.op_key:
               found=True
               break
        if found and slot.timer>wp2.target_op_limit.duration   :
            masks[Actions.top]=1
            if slot.cfg.op_key<SHARE.MIN_OP_KEY:
                masks[Actions.stay]=1
            return True
        return False

