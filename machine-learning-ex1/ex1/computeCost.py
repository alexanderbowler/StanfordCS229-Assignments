import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0
    

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    

    # ==========================================================
    for i in range(y.size):
        h = np.dot(X[i],np.transpose(theta))
        cost += (h-y[i])**2


    return cost/(2*m)
