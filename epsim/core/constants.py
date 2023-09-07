from __future__ import annotations

import numpy as np
from enum import IntEnum

class SHARE:
    START_KEY:int=101
    SWAP_KEY:int=102
    END_KEY:int=103
    MIN_OP_KEY:int=200
    TILE_SIZE:int=32
    MAX_X:int=32
    MAX_Y:int=2
    MAX_OP_TIME:int=100
    OBJ_TYPE_SIZE:int=3
    OP_TYPE1_SIZE:int=3
    OP_TYPE2_SIZE:int=6
    PRODUCT_TYPE_SIZE:int=3

WINDOW_TITLE={
    'chinese':'电镀仿真器 | 换天车(tab),换产品(q),移动(→,↑,←,↓)',
    'english':'Electroplating simulator | Change the overhead crane (tab), change the product (q), move (→,↑,←,↓)', 
}

class ObjType(IntEnum): 
    Empty = 0
    Crane = 1
    Slot = 2 
    Workpiece =3
    

'''

top
o-------->  x
|
|
↓ y 
bottom
'''
class Actions(IntEnum): 
    stay = 0
    right = 1
    top = 2 # to top
    left = 3
    bottom = 4
    
   
    
Directions = ["o","→","↑","←","↓"]
  
DIR_TO_VEC = [ 
    np.array((0, 0)),
    np.array((1, 0)),
    np.array((0, -1)),
    np.array((-1, 0)),
    np.array((0, 1)),
]



