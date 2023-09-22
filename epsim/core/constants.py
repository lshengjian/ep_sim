from __future__ import annotations

import numpy as np
from enum import IntEnum

class SHARE:
    DISPATCH_CODE:str='DP'
    START_KEY:int=101
    SWAP_KEY:int=102
    END_KEY:int=103
    MIN_OP_KEY:int=200
    MAX_OP_TIME:int=100
    OBJ_TYPE_SIZE:int=3
    OP_TYPE1_SIZE:int=3
    OP_TYPE2_SIZE:int=6
    PRODUCT_TYPE_SIZE:int=3
    MAX_X:int=100
    MAX_Y:int=2
    MAX_CRANE_SEE_DISTANCE:int=7
    MAX_OBS_LIST_LEN:int=2*MAX_CRANE_SEE_DISTANCE+1+2
    MAX_STATE_LIST_LEN:int=50
    MIN_CRANE_SAFE_DISTANCE:int=2
    TILE_SIZE:int=48

    SHORT_ALARM_TIME:int=3
    LONG_ALARM_TIME:int=20

    EPS:float=1e-4
    #AUTO_DISPATCH:bool=False
    OBSERVATION_IMAGE:bool=False




WINDOW_TITLE={
    'chinese':'电镀仿真器 | 换天车(left ctrl),换产品(left shift),选产品(space),移动(→,↑,←,↓)',
    'english':'Electroplating simulator | Change the overhead crane (left ctrl), change the product (left shift),select product(space), move (→,↑,←,↓)', 
}

class ObjType(IntEnum): 
    Empty = 0
    Crane = 1
    Slot = 2 
    Workpiece =3
    
class DispatchAction(IntEnum): 
    NOOP = 0
    NEXT_PRODUCT_TYPE = 1
    SELECT_CUR_PRODUCT = 2 
    

'''

top
o-------->  x
|
|
↓ y 
bottom
'''
class CraneAction(IntEnum): 
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



