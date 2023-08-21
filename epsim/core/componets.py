from dataclasses import dataclass as component

@component
class Idle:
    def __str__(self) -> str:
        return f'FREE'

@component
class WithJob:
    job_id:int =0
    def __str__(self) -> str:
        return f'WJ:{self.job_id}'

@component
class Count:
    steps:int = 0
    def __str__(self) -> str:
        return f'ST:{self.steps}'

@component
class Wait:
    duration: int = 0
    timer:int = 0
    @property
    def left(self):
        return self.duration-self.timer
    def __str__(self) -> str:
        return f'WT:[{self.left}]'
    
#===========================   
@component
class Slot:
    group:int =0
    code: str=''
    offset:int =0
    op_code:str='' #工艺代号
    
    
    def __str__(self) -> str:
        return f'{self.code}[{self.offset}]'#[{self.group}]

@component
class Locked: #已经安排天车来提取
    lock_id:int = 0#天车id
    move_dir:int = 0#天车运动方向
    timer:int = 0
    def __str__(self) -> str:
        return f'LOCK:{self.timer}'

# @component
# class ReqCrane: #请求天车来提取
#     def __str__(self) -> str:
#         return f'!!!'
#===========================
@component
class Crane:
    group:int=0 #分组，碰撞检测只针对同组的
    code: str = ''
    min_offset:int=0
    max_offset:int=0
    speed:int=1 #平面移动速度
    speed_up_down:int=0.5 #上下移动速度
    stop_wait:int=2 #挂载物料到达后稳定时间
    offset:int=0
    height:int=0
    
    def __str__(self) -> str:
        return f'{self.code}({self.offset:.1f})'
    def __eq__(self, other):
        return self.code == other.code
    def __hash__(self):
        return hash(self.code)

@component
class Down:
    def __str__(self) -> str:
        return f'v'
@component
class Up:
    timer:int=0
    def __str__(self) -> str:
        return f'^'
@component
class MoveTo:
    offset: int = 0
    auto:bool=True
    wait_time: int = 0
    def __str__(self) -> str:
        flag='A' if self.auto else 'M'
        return f'[{flag}]->{self.offset:.0f}'

#===========================   
@component
class JobDone: 
    pass

@component
class Job:
    code: str = ''
    proc_code:str = 'P1'
    op_index=0
    offset:float=0 #当前偏移位置
    start_time:int=0
    end_time:int=0
    def __eq__(self, other):
        return other and self.code == other.code
    def __hash__(self):
        # 注意__hash__需要返回一个整数
        return hash(self.code)
    def __str__(self) -> str:
        return f'{self.code}|{self.op_index+1}' #{self.proc_code}|

  











    






