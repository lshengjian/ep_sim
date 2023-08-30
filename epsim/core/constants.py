from __future__ import annotations

import numpy as np

from enum import IntEnum


class Actions(IntEnum):
    stay = 0
    right = 1
    down = 2 # to top
    left = 3
    up = 4
    
    
    
Directions = ["o","→","↑","←","↓"]
  
DIR_TO_VEC = [ 
    np.array((0, 0)),
    np.array((1, 0)),
    np.array((0, -1)),# （0，0） at left top
    np.array((-1, 0)),
    np.array((0, 1)),
    
    
]



