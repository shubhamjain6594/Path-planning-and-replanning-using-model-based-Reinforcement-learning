import numpy as np
from gym.envs.toy_text import discrete
 
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld(discrete.DiscreteEnv):
    def __init__(self, shape, x, y):
        BlockState = []
        
        for i in range(len(x)):
           BlockState.append(shape[0]*y[i] + x[i])
        self.nA = 4
        self.nS = shape[0]*shape[1]
        self.shape = shape
                
        xLimit = shape[1]
        yLimit = shape[0]
        
        nA = self.nA
        nS = self.nS
        
        P = {}
        
        gridMap = np.arange(nS).reshape(shape)
        iteration = np.nditer(gridMap, flags =['multi_index'])
        
        while not iteration.finished:
            s = iteration.iterindex
            y, x = iteration.multi_index
            
            if s in BlockState:
                iteration.iternext()
                continue
            
            P[s] = {a: [] for a in range(nA)}
            
            is_done = lambda s: s ==0 or s==(nS-1)
            reward = 0.0 if is_done(s) else -1.0
            
            if is_done(s):
                
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
                
            else:
                for BlkInd, BlkSt in enumerate(BlockState):
                
                    s_up = s if (((s-xLimit) == BlkSt) or y==0) else (s - xLimit)
                    s_right = s if (((s+1) == BlkSt) or x == (xLimit-1)) else (s+1)
                    s_down = s if (((s+xLimit) == BlkSt) or y == yLimit - 1) else (s + xLimit)
                    s_left = s if (((s-1) == BlkSt) or x == 0) else (s - 1)
                    
                    P[s][UP] = [(1.0, s_up, reward, is_done(s_up))]
                    P[s][RIGHT] = [(1.0, s_right, reward, is_done(s_right))]
                    P[s][DOWN] = [(1.0, s_down, reward, is_done(s_down))]
                    P[s][LEFT] = [(1.0, s_left, reward, is_done(s_left))]
                    
            iteration.iternext()
            
        isd = np.ones(nS)/(nS)
        self.P = P
        self.isd = isd
        
        super(GridWorld, self).__init__(nS, nA, P, isd)