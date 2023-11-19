import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================
    #y = y.reshape((m,1))
    h = sigmoid(np.matmul(X,theta))
    #h = h.reshape((h.size,1))
    #print(h)
    cost = (np.matmul((-y).T,np.log(h)) - np.matmul((np.add(-y,1)).T,np.log(np.add(-h,1))))/m
    #print(cost)
    #print(np.subtract(h,y).shape)
    grad = np.matmul((np.subtract(h,y)).T,X)/m
    #print(grad)
    return cost, grad
