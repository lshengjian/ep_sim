from typing import List, Optional
import numpy as np
from gymnasium.utils import seeding
from .config import *

class World:
    def __init__(self,desc:np.ndarray):
        
        self.desc=desc
        
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.nA = nA = 5
        self.nS = nS = nrow * ncol
        self.state=0
        self.lastaction=None

        self.initial_state_distrib = np.array(desc == b'S').astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}#动作转移概率
        for row in range(nrow):
            for col in range(ncol):
                s =self.idx2state(row, col)
                for a in range(nA):
                    li = self.P[s][a]
                    li.append((1.0, *self.try_move(row, col, a)))

    def try_move(self,row:int, col:int, action:int):
        ok,newrow, newcol = self.next(row, col, action)
        newstate =self.idx2state(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        terminated = bytes(newletter) in b"G"
        reward=0
        if not ok:
            reward=OUT_BOUND
        elif newletter == b"G":
            reward = IN_GOAL
        elif newletter == b"X":
            reward = IN_FORBIDDEN
        return newstate, reward,terminated

    
    def idx2state(self,row, col):
        return row*self.ncol+col
    
    def state2idx(self,state):
        return state//self.ncol,state%self.ncol
    
    def next(self,row, col, a):
        canPass=True
        if a == LEFT:
            if col - 1< 0:canPass=False
            col = max(col - 1, 0)
        elif a == DOWN:
            if row + 1> self.nrow - 1:
                canPass=False
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            if col + 1> self.ncol - 1:
                canPass=False
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            if row - 1< 0:canPass=False
            row = max(row - 1, 0)
        return canPass,row, col