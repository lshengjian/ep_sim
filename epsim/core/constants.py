from __future__ import annotations

import numpy as np

from enum import IntEnum

WINDOW_TITLE={
    'chinese':'电镀仿真器 | 换天车(tab),换产品(q),移动(→,↑,←,↓)',
    'english':'Electroplating simulator | Change the overhead crane (tab), change the product (q), move (→,↑,←,↓)', 
}
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



