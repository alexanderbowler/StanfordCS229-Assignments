import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================
    h = sigmoid(np.matmul(X,theta))
    cost = (np.matmul((-y).T,np.log(h)) - np.matmul((np.add(-y,1)).T,np.log(np.add(-h,1))))/m + lmd/(2*m)*theta.T.dot(theta)
    
    grad = np.matmul((np.subtract(h,y)).T,X)/m
    add = lmd/m*theta
    add[0] = 0
    #print(add)
    grad+=add
    return cost, grad
