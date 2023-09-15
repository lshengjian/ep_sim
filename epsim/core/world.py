from __future__ import annotations

import math
from typing import  List,Dict
from collections import defaultdict
from .constants import *
from .componets import *
from .world_object import  *
from .constants import Actions
from .crane import Crane
from .slot import Slot
from .workpiece import Workpiece
from .config import build_config
from epsim.utils import get_state,get_observation # for image
import logging
logger = logging.getLogger(__name__)
'''
外部接口
1) 加入一批物料
add_jobs(codes:List[str]) #add_jobs(['A','A','B'])
2) 选择下一个天车为当前控制对象
next_crane()
3) 给当前天车发作业指令
set_command(action:int]) #set_command(0)
4) 给所有天车发作业指令
set_commands(actions:List[int]) #set_command([0,1])
5) 更新系统
update()
6) 获取当前天车的状态
get_observation()
7) 获取全部天车的状态
get_observations()

自动版规则：
1) 当上料槽位有空时,顺序把待加工物料放到槽位上
2) 当物料放入下料槽位后,物料加工结束,从系统内消失。

加分规则：
1) 在加工槽位出槽时加工的时间在正常加工时段,加1分
2) 物料正常加工结束,加10分

扣分规则：
1) 在加工槽位出槽时物料加工的时间少于最少或多余最多3秒内  减1分
2) 在加工槽位出槽时物料加工的时间少于最少或多余最多超3秒 减3分
3) 放错槽位 游戏结束
4) 天车相撞 游戏结束
'''
class World:
    def __init__(self,config_directory='test01',auto_dispatch=False):
        
        self.config_directory=config_directory
        self.enableAutoPut=auto_dispatch
        self.is_over=False
        self.todo_cnt=0
        self.rewards={}
        self.score=0
        self.step_count=0
        self.ops_dict:Dict[int,OperateData]=None
        
        self.max_y: int = 2
        self.all_cranes:List[Crane] = []
        self.pos_slots:Dict[int,Slot] = {} #Start,End,Belt,Tank
        self.group_slots:Dict[int,List[Slot]]=defaultdict(list)
        self.group_limits:Dict[int,List[int,int]]={}
        self.group_cranes:Dict[int,List[Crane]]=defaultdict(list)
        self.starts:List[Slot]=[]
        self.ends:List[Slot]=[]
        self.product_procs:Dict[str,List[OpLimitData]]=defaultdict(list)
        self.products=[]
        self.prd_idx=0
        self.cur_crane_index=0
        self.state=None
        self.max_steps=1000
        self.masks={}
        self.load_config()
    
    @property
    def  cur_crane(self):
        return self.all_cranes[self.cur_crane_index]

    def add_jobs(self,ps:List[str]=[]):
        self.products.extend(ps)
        self.todo_cnt+=len(ps)
        #print('add_jobs')
        if self.enableAutoPut:self.products2starts()

    def next_crane(self):
        for c in self.all_cranes:
            c.color=Color(255,255,255)
        self.cur_crane_index=(self.cur_crane_index+1)%len(self.all_cranes)
        self.cur_crane.color=Color(255,0,0)
        
    def put_product(self):
        if self.cur_prd_code is None:
            return
        self.products2starts()
        # wp=Workpiece.make_new(self.cur_prd_code)
        # self.plan_next(wp)
        # start=self.get_free_slot(1,wp)
        # if start is None:
        #     return
        # start.put_in(wp)

    def shift_product(self):
        cur_prd_code=None if len(self.products)<1 else self.products[0]
        
        if cur_prd_code is None:
            self.cur_prd_code=None
            return 
        buff1=[]
        buff2=[]
        
        for p in self.products:
            if p==cur_prd_code:
                buff1.append(p)
            else:
                buff2.append(p)
        buff2.extend(buff1) 
        self.products=buff2
        self.cur_prd_code=buff2[0]
        
    def set_command(self,action:Actions):
        crane=self.all_cranes[self.cur_crane_index]
        crane.set_command(action)

    def set_commands(self,actions:List[Actions]):
        assert len(actions)==len(self.all_cranes)
        for i,crane in enumerate(self.all_cranes):
            crane.set_command(actions[i])
    
    def update(self):
        if self.is_over:
            return
        self.step_count+=1
        if self.step_count>self.max_steps:
            self.is_over=True
            return


        for crane in self.all_cranes:
            self.rewards[crane.cfg.name]=0
            crane.step()

        slots=self.pos_slots.values()
        for slot in slots:
            slot.step()
        self._check_cranes()
        self._check_slots()
        self.score+=sum(self.rewards.values())
        if self.todo_cnt<1:
            self.is_over=True
            logger.info("!!!OK!!!")
        elif self.enableAutoPut:
            have_empty=False
            for s in self.starts:
                if s.carrying is None:
                    have_empty=True
                    break
            if have_empty and len(self.products)>0 and  np.random.random()<0.001:
                self.products2starts()
    
    def get_state_img(self,scrern_img:np.ndarray,nrows,ncols): #仿真系统的全部状态数据
        self.state= get_state(scrern_img,nrows,ncols,SHARE.TILE_SIZE)
        return self.state
    
    def get_observation_img(self,agv:Crane):
        return  get_observation(self.state,agv.x,SHARE.MAX_AGENT_SEE_DISTANCE,SHARE.TILE_SIZE,SHARE.MAX_X)

    def get_state(self): #仿真系统的全部状态数据
        rt=[]
        for crane in self.all_cranes:
            rt.append(crane.state2data())
        for slot in self.pos_slots.values:
            rt.append(slot.state2data()) 
        for pcode in self.products:
            wp=Workpiece.make_new(pcode)
            rt.append(wp.state2data())
        return np.array(rt,dtype=np.float32)
    
    def get_observation(self,agv:Crane):

   
        group:int=agv.cfg.group
        rt=[]
              
        rt.append(agv.state2data())
        for crane in self.group_cranes[group]:
            if agv==crane:continue
            rt.append(crane.state2data())
        cs=[]
        for slot in self.group_slots[group]:
            dis=abs(slot.x-agv.x)
            if dis<=SHARE.MAX_AGENT_SEE_DISTANCE:
                cs.append((dis,slot))
        if len(cs)>0:
            cs.sort(key=lambda x:x[0])
            size=len(rt)
            for _,slot in cs:
                if size<SHARE.MAX_OBS_LIST_LEN:
                    rt.append(slot.state2data())
                    size+=1
                else:
                    break 
                              
        for k in range(len(rt),SHARE.MAX_OBS_LIST_LEN):
            rt.append([0.]*len(rt[0]))
        rt=np.array(rt,dtype=np.float32)
        #print(rt.shape)
        return rt.ravel()

    def reset(self):
        #self.load_config()
        self.is_over=False
        self.todo_cnt=0
        self.cur_crane_index=0
        self.prd_idx=0
        self.cur_prd_code=None
        self.step_count=0
        self.score=0
        self.products.clear()
        for crane in self.all_cranes:
            crane.reset()
        for slot in  self.pos_slots.values():
            slot.reset()
        Workpiece.UID.clear()
        for crane in self.all_cranes:
            self.rewards[crane.cfg.name]=0
        
    def get_crane_bound(self,crane:Crane)->Tuple[Slot|Crane,Slot|Crane]:
        g=crane.cfg.group
        x1,x2=map(int, self.group_limits[g])
        s1=self.pos_slots[x1]
        s2=self.pos_slots[x2]
        l_side=list(filter(lambda agv:agv.x<crane.x, self.group_cranes[g]))
        r_side=list(filter(lambda agv:agv.x>crane.x, self.group_cranes[g]))
        l_side.sort(key=lambda c:c.x,reverse=True)
        r_side.sort(key=lambda c:c.x)
        left=s1 if len(l_side)<1 else l_side[0]
        right=s2 if len(r_side)<1 else r_side[0]
        return left,right
            
    # def get_group_bound(self,group=1):
    #     return self.group_limits[group]
    
    def products2starts(self):
        ps=self.products[:]
        
        for s in self.starts:
            #print(s)
            if not s.locked and len(ps)>0:
                wp:Workpiece=Workpiece.make_new(ps[0],s.x)
                #print('products2starts')
                self.plan_next(wp)
                s.put_in(wp)
                ps.remove(ps[0])
                break
        self.products=ps
        self.cur_prd_code=None if len(ps)<1 else ps[0]

    def load_config(self):
        self.ops_dict,slots,cranes,procs=build_config(self.config_directory)
        self.group_limits.clear()
         
        self.all_cranes.clear()
        self.group_cranes.clear()

        self.pos_slots.clear()
        self.group_slots.clear()
                
        self.starts.clear()
        self.ends.clear()
        self.product_procs.clear()


        for data in procs:
            pd:OpLimitData=data
            self.product_procs[pd.product_code].append(pd)

        for data in slots:
            s:SlotData=data
            g=s.group
            x1,x2=self.group_limits.get(g,[1000,-1000])
            for x in s.offsets:
                if x<x1:
                    x1=x
                if x>x2:
                    x2=x 
                self.group_limits[g]=[int(x1),int(x2)]   
                slot:Slot=Slot(x,s)
                slot.color=self.ops_dict[slot.cfg.op_key].color
                if s.op_key==SHARE.START_KEY:
                    self.starts.append(slot)
                elif s.op_key==SHARE.END_KEY:
                    self.ends.append(slot)
                self.pos_slots[int(x)]=slot
                self.group_slots[g].append(slot)

        for data in cranes:
            cfg:CraneData=data
            crane:Crane=Crane(cfg.offset,cfg)
            self.all_cranes.append(crane)
            self.group_cranes[cfg.group].append(crane)
        self.max_x: int = max(list(self.pos_slots.keys()))
 
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

          




    def _not_out_bound(self, crane, masks,left,right):
        eps=SHARE.EPS
        if crane.y<eps:
            masks[Actions.top]=0
        elif crane.y>(self.max_y-eps):
            masks[Actions.bottom]=0
        if eps<crane.y<(self.max_y-eps):
            masks[Actions.left]=0
            masks[Actions.right]=0
        
        safe_dis=SHARE.MIN_AGENT_SAFE_DISTANCE+1

        if type(left) is Crane :
            if crane.x<=left.x+safe_dis:
                masks[Actions.left]=0
        elif crane.x==left.x:
            masks[Actions.left]=0
        
        if type(right) is Crane :
            if crane.x>=right.x-safe_dis:
                masks[Actions.right]=0
        elif crane.x==right.x:
            masks[Actions.right]=0

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


           
        


    def _get_next_op_limit(self,wp:Workpiece):
        slot:Slot=wp.attached
        assert slot  is not None
        cur=wp.target_op_limit
        ps=self.product_procs[wp.product_code]
        cur_idx=ps.index(wp.target_op_limit)
        rt=None
        if cur_idx<len(ps)-1:
            rt=ps[cur_idx+1]
        return rt

        # 如果下个工序是交换，则一定要靠交换位的天车才能来提取


                
    def mask2str(self,masks):
        flags=[]
        for i,m in enumerate(masks):
            if m: flags.append(Directions[i])
        return ''.join(flags)

    def get_masks(self,crane:Crane):
        masks = np.zeros(5, dtype=np.int8)
        
        if self._limit_only_up(crane,masks) or self._limit_only_down(crane,masks):
            self.masks[crane.cfg.name]=masks
            return masks

        left,right=self.get_crane_bound(crane)
        if self._should_up(crane,masks,left,right):
            self.masks[crane.cfg.name]=masks
            #self._not_out_bound(crane,masks,left,right)
            return masks
        if self._should_down(crane,masks):
            self.masks[crane.cfg.name]=masks
            return masks
        if self._should_move(crane,masks,left,right):
            self._not_out_bound(crane,masks,left,right)
            self.masks[crane.cfg.name]=masks
            if sum(masks)<1:
                masks[Actions.stay]=1
            return masks

        masks = np.ones(5, dtype=np.int8)
        masks[Actions.top]=0
        masks[Actions.bottom]=0
        self._not_out_bound(crane,masks,left,right)
        self.masks[crane.cfg.name]=masks
        #print('random walk')
        return masks


    

    def plan_next(self,wp:Workpiece):
        ps=self.product_procs[wp.product_code]
        #print(ps[0])
        if wp.target_op_limit is None:#第一次规划，放到上料位
            wp.set_next_operate(ps[0],self.ops_dict)
            return
        idx=ps.index(wp.target_op_limit)
        if idx<len(ps)-1:
            wp.set_next_operate(ps[idx+1],self.ops_dict)

    def get_free_slot(self,group:int,wp:Workpiece)->Slot:
        '''
        获取指定区间组的指定工艺的空闲工作槽位
        '''
        slots = self.group_slots[group]
        data=[]
        for s in slots:
            #print(s)
            if s.locked==False and s.cfg.op_key==wp.target_op_limit.op_key and s.carrying==None:
                data.append((abs(wp.x-s.x),s))
        if len(data)<1:
            return None
        data.sort(key=lambda x:x[0])
        data[0][1].locked=True
        return data[0][1]
    
  

    def _translate(self,source:Crane|Slot,target:Crane|Slot):
        '''
        在天车和工作槽位间转移物料
        '''  

        wp,reward=source.take_out()
        if type(target) is Slot: 
            if target in self.ends:
                self.rewards[source.cfg.name]+=10
                self.todo_cnt-=1
                del wp
                return 
            if target.cfg.op_key==SHARE.SWAP_KEY:
                x=target.x+1
                target=self.pos_slots[x]
        if target.carrying!=None:
            logger.info(f'{target} already have {target.carrying}')
            self.rewards[source.cfg.name]-=5
            self.is_over=True
            return 

        target.put_in(wp)
        if type(target) is Crane:
            self.rewards[target.cfg.name]=reward
            self.plan_next(wp)
        # if source in self.starts and self.enableAutoPut and np.random.random()<0.001:
        #     self.products2starts()
         
        



    def _check_collide(self,crane:Crane)->bool:
        '''
        天车移动后进行碰撞检测
        '''  
        collide=False
        pos=self._round(crane.x)
        slot=self.pos_slots.get(pos,None)
        if slot!=None and abs(crane.y-1)<0.1:
            wp:Workpiece=crane.carrying
            if wp==None and crane.last_action==Actions.top and slot.carrying!=None:
                self._translate(slot,crane)
                
            if wp!=None and crane.last_action==Actions.bottom  :
                if wp.target_op_limit.op_key!=slot.cfg.op_key:
                    logger.info(f'{wp.target_op_limit.op_key} not same as {slot.cfg.op_key}')
                    self.rewards[crane.cfg.name]-=5
                    self.is_over=True
                    return
                self._translate(crane,slot)
                
                    
        cranes=self.group_cranes[crane.cfg.group]
        for c in cranes:
            if c==crane:
                continue
            if abs(c.x-crane.x)<SHARE.MIN_AGENT_SAFE_DISTANCE:
                collide=True
                logger.error(f'{c} too close to {crane}')
                if c.last_action!=Actions.stay:
                    self.rewards[c.cfg.name]-=5
                if crane.last_action!=Actions.stay:
                    self.rewards[crane.cfg.name]-=5
                break

        return collide
        



    def _check_slots(self):
        slots=self.pos_slots.values()
        for s in slots:
            if s.cfg.op_key<SHARE.MIN_OP_KEY or s.carrying is None:
                continue
            op:OpLimitData=s.carrying.target_op_limit
            if s.timer>op.max_time+SHARE.LONG_ALARM_TIME:
                self.is_over=True
                logger.error(f'{s} op timeout!')
                agvs=[]
                for agv in self.group_cranes[s.cfg.group]:
                    if (agv.carrying is None):
                        agvs.append((abs(agv.x-s.x),agv))
                
                if len(agvs)>0:
                    agvs.sort(key=lambda x:x[0])
                    agv=agvs[0][1]  #最近的空闲天车失职！
                    self.rewards[agv.cfg.name]-=5
                    

                break  

    def _check_cranes(self):
        for c in self.all_cranes:
            if  self._out_bound(c):
                logger.error(f'{c} out  bount!')
                self.rewards[c.cfg.name]-=5
                self.is_over=True
                
                return
            if self._check_collide(c):
                self.is_over=True
                #logger.error(f'{c} collided!')
                return
                
    def _out_bound(self,crane:Crane):
        x=crane.x
        y=crane.y
        x1,x2=self.group_limits[crane.cfg.group]
        return x<x1 or y<0 or x>x2 or y>self.max_y
    
    def _round(self,x:float)->int:
        return int(x+0.5)

        


    
    def pprint(self):
        for k,cs in self.group_cranes.items():
            print(f'Group {k}')
            for c in cs:
                print(f'{c}')
        print(f'='*18)
        for k,cs in self.group_slots.items():
            print(f'Group {k}')
            for c in cs:
                print(f'{c}')

'''




'''



   





 
                












 

 