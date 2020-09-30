import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

# input
# a is the msg sending window
a = np.array([[0,1,1,1,1,1,0,0,0,0],
                [1,1,1,1,0,0,0,0,0,0],
                [0,1,1,1,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1],
                [0,1,1,1,1,1,1,1,0,0],
                [0,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,1,1,1,1,0,0]])

# m is the number of msgs to be sent
m = np.array([40000,50000,40000,45000,60000,80000,70000,30000])

r = len(a[:,1])
c = len(a[1,:])

# constraint is sum of msgs sent in intervals equals number of msgs to be sent
def constructConstraint(a,m):
    aVectorArr = []
    for i in range(r):
        aVector = np.zeros(r*c)
        for j in range(c):
            aVector[i*c+j] = a[i][j]
        aVectorArr.extend(aVector)
    aVectorArr = np.reshape(aVectorArr, (r, r*c))
    linear_constraint = LinearConstraint(aVectorArr,m,m)
    return linear_constraint

# bound of msg sent in interval is [0,0] or [0,m[i]]
def constructBound(a,m):
    r = len(a[:,1])
    c = len(a[1,:])
    minArr = np.zeros(r*c)
    maxArr = []
    for i in range(r):
        for j in range(c):
            maxArr.append(m[i]*a[i][j])
    return Bounds(minArr,maxArr)

def objFn(x):
    x = np.reshape(x, (r, c))
    cSumArr = np.sum(x,axis=0)
    res = 0
    # sum of square of diff in column sum, better performance than max(cSumArr)
    for i in range(c):
        for j in range(i+1,c):
            res += (cSumArr[i] - cSumArr[j])**2
    return res

linear_constraint = constructConstraint(a,m)
bounds = constructBound(a,m)
x0 = a
x0 = x0.flatten()
res = minimize(objFn, x0, method='trust-constr',
                constraints=[linear_constraint],
                options={'maxiter': 500, 'verbose': 1},bounds=bounds)

res = np.reshape(res.x,(r,c))
res = np.rint(res)
print(res)
cSum = np.sum(res,axis=0)
print(cSum)