from __future__ import annotations

import numpy as np

from enum import IntEnum

'''

top
o-------->  x
|
|
↓ y 
bottom
'''
class Actions(IntEnum): #顺时针
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



