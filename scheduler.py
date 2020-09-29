import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

#input
a = np.array([[0,1,1,1,1,1,0,0],
                [1,1,1,1,0,0,0,0],
                [0,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,0],
                [0,0,1,1,1,1,1,1],
                [0,0,0,1,1,1,0,0]])

m = np.array([400,500,400,450,600,300])

r=len(a[:,1])
c=len(a[1,:])

def constructConstraint(a,m):
    aVectorArr=[]
    for i in range(r):
        aVector=np.zeros(r*c)
        for j in range(c):
            aVector[i*c+j]=a[i][j]
        aVectorArr.extend(aVector)
    aVectorArr = np.reshape(aVectorArr, (r, r*c))
    linear_constraint = LinearConstraint(aVectorArr,m,m)
    print(aVectorArr)
    return linear_constraint

def constructBound(a,m):
    r=len(a[:,1])
    c=len(a[1,:])
    minArr=np.zeros(r*c)
    maxArr=[]
    for i in range(r):
        for j in range(c):
            maxArr.append(m[i]*a[i][j])
    print(maxArr)
    return Bounds(minArr,maxArr)

linear_constraint=constructConstraint(a,m)
bounds = constructBound(a,m)
x0 = a


def objFn(x):
    x = np.reshape(x, (r, c))
    cSumArr=[]
    for j in range(c):
        cSum=0
        for i in range(r):
            cSum+=a[i][j]*x[i][j]
        cSumArr.append(cSum)
    res=0
    # sum of square of diff in column sum
    for i in range(c):
        for j in range(i+1,c):
            res+=(cSumArr[i]-cSumArr[j])**2
    return res

x0 = x0.flatten()
res = minimize(objFn, x0, method='trust-constr',
                constraints=[linear_constraint],
                options={'maxiter': 500, 'verbose': 1},bounds=bounds)
#print(res)
res=np.rint(res.x)
res=np.reshape(res,(r,c))
print(res)