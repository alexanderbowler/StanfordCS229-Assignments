import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]

    # Return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned logistic regression parameters.
    #                You should set p to a 1D-array of 0's and 1's
    #


    # ===========================================================
    #print(sigmoid(X.dot(theta)))
    p = np.where(sigmoid(X.dot(theta))>0.5,1,0)
    print(p)
    

    return p
