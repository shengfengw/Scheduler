import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import json

prevSchedule = np.zeros(10);

# class MessageGroup:
#   def __init__(self, name, sndWindow, msgNum, capacity):
#     self.name = name
#     self.sndWindow = sndWindow
#     self.msgNum = msgNum
#     self.capacity = capacity

class ScheduleBatch:
  def __init__(self, jsonFilePath):
    # construct input
    # A is the msg sending window
    self.W = constructW(jsonFilePath)
    # M is the number of msgs to be sent
    self.M = constructM(jsonFilePath)
    # C is the capacity of msgs
    self.C = constructC(jsonFilePath)


def constructW(jsonFilePath):
    with open(jsonFilePath) as f:
      data = json.load(f)
    W = []
    for item in data:
        W.append(item["sndWindow"])
    return np.array(W)

def constructM(jsonFilePath):
    with open(jsonFilePath) as f:
      data = json.load(f)
    M = []
    for item in data:
        M.append(item["msgNum"])
    return np.array(M)

def constructC(jsonFilePath):
    with open(jsonFilePath) as f:
      data = json.load(f)
    C = []
    for item in data:
        C.append(item["capacity"])
    return np.array(C)

# constraint is sum of msgs sent in intervals equals number of msgs needed to be sent, dimension (r, r*c)
def constructConstraint(W,M):
    r = len(W[:,1])
    c = len(W[1,:])
    wVectorArr = []
    for i in range(r):
        wVector = np.zeros(r*c)
        for j in range(c):
            wVector[i*c+j] = W[i][j]
        wVectorArr.extend(wVector)
    wVectorArr = np.reshape(wVectorArr, (r, r*c))
    linear_constraint = LinearConstraint(wVectorArr,M,M)
    return linear_constraint

# bound of msg sent in interval is [0,0] or [0,m[i]]
def constructBound(W,M,C):
    r = len(W[:,1])
    c = len(W[1,:])
    minArr = np.zeros(r*c)
    maxArr = []
    for i in range(r):
        for j in range(c):
            maxArr.append(min(M[i],C[i])*W[i][j])
    return Bounds(minArr,maxArr)

def generateGuess(W,M,C):
    r = len(W[:,1])
    c = len(W[1,:])
    x0 = []
    windowSize = np.sum(W,axis=1)
    for i in range(r):
        for j in range(c):
            # guess should be total msg num divided by window size, not exceeding total msg num
            x0.append(min(M[i]/windowSize[i],C[i])*W[i][j])
    return np.array(x0).flatten()

def objFn(x,W):
    r = len(W[:,1])
    c = len(W[1,:])
    x = np.reshape(x, (r, c))
    currSchedule = np.add(prevSchedule,np.sum(x,axis=0))
    # obj is var
    return np.var(currSchedule)

def scheduleAct(jsonFilePath,prevSchedule):
    scheduleBatch = ScheduleBatch(jsonFilePath)
    # print out input matrix
    W = scheduleBatch.W
    M = scheduleBatch.M
    C = scheduleBatch.C
    print("W:")
    print(W)
    print("M:")
    print(M)
    print("C:")
    print(C)

    r = len(W[:,1])
    c = len(W[1,:])

    linear_constraint = constructConstraint(W,M)
    bounds = constructBound(W,M,C)
    x0 = generateGuess(W,M,C)
    res = minimize(objFn, x0, W, method='trust-constr',
                    constraints=[linear_constraint],
                    options={'maxiter': 300, 'verbose': 0},bounds=bounds)
    x = np.reshape(res.x,(r,c))
    x = np.rint(x)
    print("x:")
    print(x)
    cSum = np.sum(x,axis=0)
    print("batchSum:")
    print(cSum)
    return x

def updatePS(x):
    global prevSchedule
    prevSchedule = np.add(prevSchedule,np.sum(x,axis=0))
    return prevSchedule

for i in range(3):
    print("==================== Batch #"+str(i)+": =======================")
    x = scheduleAct("input"+str(i)+".json",prevSchedule)
    updatePS(x)
    print("currSchedule:")
    print(prevSchedule)

