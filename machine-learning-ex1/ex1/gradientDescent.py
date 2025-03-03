import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # Hint: X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )


        # ===========================================================
        # Save the cost every iteration
        sum=0
        for i in range(y.size):
            h = np.dot(X[i],np.transpose(theta))
            sum += (h-y[i])*X[i]
        sum = sum/m
        theta -= alpha*sum
        #print(compute_cost(X, y, theta))
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        h=np.dot(X,np.transpose(theta))
        hy = np.array(h-y)[np.newaxis]
        theta -= alpha*1/m*np.sum((np.transpose(hy))*X,axis=0)
        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
    