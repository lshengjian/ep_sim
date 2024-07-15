from __future__ import annotations
from dataclasses import dataclass as component
from typing import Tuple


    
@component
class Color:
    r:int = 0
    g:int = 0
    b:int = 0
    a:int = 255
    @property
    def rgb(self):
        return self.r,self.g,self.b
    @property
    def rgba(self):
        return self.r,self.g,self.b,self.a
    def __str__(self) -> str:
        return f'{self.rgb}'
@component
class State:
    obj_type:int=0 #1--天车 2--加工槽位 3--物料
    op_key:int = 0
    product_code:str = ''
    x:float = 0.
    y:float = 0.
    op_duration:int=0
    op_time:int=0

    def clone(self) -> State:
        return State(self.obj_type, self.op_key, self.product_code, \
                     self.x, self.y, \
                     self.op_duration, self.op_time   )
@component
class Index:
    id:int =0
    def __eq__(self, other):
        return self.id == other.id
    def __hash__(self):
        return hash(self.id)

@component
class OperateData(Index):
    key:int=0
    name:str=''
    color:Color=Color()
    def __str__(self) -> str:
        return f'[{self.key}]{self.name} {self.color}'

@component
class SlotData(Index):
    group:int=1
    op_key:int=0
    offsets:Tuple=tuple()
    
    op_name:str=''
    def __str__(self) -> str:
        xs='|'.join(map(str,self.offsets))
        return f'[{self.group}] {self.op_name} {xs}'
    
@component
class CraneData(Index):
   group:int=0
   name:str=''
   offset:int=0.0
   speed_x:float=1.0
   speed_y:float=1.0
   def __str__(self) -> str:
        return f'[{self.group}] {self.name} {self.offset} ({self.speed_x},{self.speed_x})'
   
@component
class OpLimitData(Index):
   product_code:str=''
   op_key:int=0
   min_time:int=0
   max_time:int=0
   
   op_name:str=''
   @property
   def duration(self):
       return (self.max_time+self.min_time)//2
   def __str__(self) -> str:
        return f'[{self.product_code}] {self.op_name} {self.min_time}->{self.max_time}'