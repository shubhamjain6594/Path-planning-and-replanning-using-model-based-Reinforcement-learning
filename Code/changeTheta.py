from policy_improvement import policy_improvement_fun
import time
from gridEnv import GridWorld
import numpy as np

shape = [30,30]

AvgCount = 5
THETA = 1

x = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7]
y = [7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16]

timeTaken1 = []
timeTaken2 = []
timeTaken3 = []
BlockStateini = []
for i in range(len(x)):
    BlockStateini.append(shape[0]*y[i] + x[i])
Grid = GridWorld(shape, x, y)

for l in range(AvgCount):
    x = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7]
    y = [7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16,7,8,9,10,11,12,13,14,15,16]
    THETA = THETA/10
    #timeTaken[l] = []
    Grid = GridWorld(shape, x, y)
    value = np.zeros(Grid.nS)
    policy = np.ones([Grid.nS, Grid.nA]) / Grid.nA

    for k in range(3):
        BlockState = []
        for i in range(len(x)):
            BlockState.append(shape[0]*y[i] + x[i])

        start = time.time()
        
        policy, val, count = policy_improvement_fun(Grid, value, BlockState, policy, theta = THETA)
        
        end = time.time()
        if k==0:
            timeTaken1.append(end-start)
            print(end-start)
        if k==1:
            timeTaken2.append(end-start)
            print(end-start)
        if k==2:
            timeTaken3.append(end-start)
            print(end-start)
        
        x += np.ones((len(x)))
        y += np.ones((len(y)))

        cnt = 0
        for m in range(len(BlockState)):
            if BlockState[m] in BlockStateini:
                cnt +=1
        print(len(x) - cnt)

timeTaken1 = np.reshape(timeTaken1, (AvgCount, int(len(timeTaken1)/AvgCount)))
timeTaken2 = np.reshape(timeTaken2, (AvgCount, int(len(timeTaken2)/AvgCount)))
timeTaken3 = np.reshape(timeTaken3, (AvgCount, int(len(timeTaken3)/AvgCount)))

print(timeTaken1)
print(timeTaken2)
print(timeTaken3)
