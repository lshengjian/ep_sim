from __future__ import annotations

import numpy as np

from enum import IntEnum


class Actions(IntEnum):
    stay = 0
    forward = 1
    down = 2
    back = 3
    up = 4
    
Directions = ["o","→","↓","←","↑"]
  
DIR_TO_VEC = [ 
    np.array((0, 0)),
    np.array((1, 0)),
    np.array((0, 1)),
    np.array((-1, 0)),
    np.array((0, -1)),
]# （0，0） at left top

'''
EMPTY="empty"
BUFF_START="start" #上料位
BUFF_END="end"     #下料位
BUFF_SWAP="swap"   #两个区间的交换位，同时属于两个区间组
TANK="tank"        #加工位：电镀、除油、水洗、烘干等
CRANE="crane"
WORKPIECE="workpiece" #工件，待电镀的物料


# Map of object type to integers
OBJECT_TO_IDX = {
    EMPTY: 0,
    BUFF_START: 1,
    TANK: 2,
    BUFF_SWAP: 3,
    BUFF_END: 4,
        
    WORKPIECE: 5,
    CRANE: 6,
    
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
'''


