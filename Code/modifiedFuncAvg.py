from policy_improvement import policy_improvement_fun
import time
from gridEnv import GridWorld
import numpy as np
import matplotlib.pyplot as plt

shape = [30,30]

#Change to 1 for non-averaged vaules
AvgCount = 100
THETA = 0.1

# Define Obstacle space here
x = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7]
y = [7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16]

timeTaken = []
BlockStateini = []
for i in range(len(x)):
    BlockStateini.append(shape[0]*y[i] + x[i])
Grid = GridWorld(shape, x, y)

for l in range(AvgCount):
    x = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7]
    y = [7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16]

    Grid = GridWorld(shape, x, y)
    value = np.zeros(Grid.nS)
    policy = np.ones([Grid.nS, Grid.nA]) / Grid.nA
    
    for k in range(12):
        BlockState = []
        for i in range(len(x)):
            BlockState.append(shape[0]*y[i] + x[i])
        
        start = time.time()
        
        policy, val, count = policy_improvement_fun(Grid, value, BlockState, policy, theta = THETA)

        if k == 0:
            value = val
        
        end = time.time()
        timeTaken.append(end-start)
        print(end-start)
        x += np.ones((len(x)))
        # Uncomment the following for diagonal movement
        # y += np.ones((len(y)))
    
        cnt = 0
        for m in range(len(BlockState)):
            if BlockState[m] in BlockStateini:
                cnt +=1
        print(len(x) - cnt)

timeTaken = np.reshape(timeTaken, (AvgCount, int(len(timeTaken)/AvgCount)))
avgTime = np.mean(timeTaken, axis = 0)
print(avgTime)

plt.figure()
plt.semilogy(avgTime)
plt.xlabel('Diagonal Displacement')
plt.ylabel('Time (log)')
plt.show()