# CS229 Machine Learning Assignments in Python
# Assignments Taken from this page Solutions from author in folder titled Solutions, Solutions in Alex Solutions are Mine
[![Assignment-author](https://img.shields.io/badge/author-nsoojin-red.svg)](https://www.linkedin.com/in/soojinro) [![python](https://img.shields.io/badge/python-3.6-blue.svg)]() [![license](https://img.shields.io/github/license/mashape/apistatus.svg)]() [![contribution](https://img.shields.io/badge/contribution-welcome-brightgreen.svg)]()

![title_image](title_image.png)

## About
If you've finished the amazing introductory Machine Learning on Coursera by Prof. Andrew Ng, you probably got familiar with Octave/Matlab programming. With this repo, you can re-implement them in Python, step-by-step, visually checking your work along the way, just as the course assignments.

## How to start
### Dependencies
This project was coded in Python 3.6
* numpy
* matplotlib
* scipy
* scikit-learn
* scikit-image
* nltk

### Installation
The fastest and easiest way to install all these dependencies at once is to use [Anaconda](https://www.continuum.io/downloads).


## Important Note
There are a couple of things to keep in mind before starting.
* all column vectors from octave/matlab are flattened into a simple 1-dimensional ndarray. (e.g., y's and thetas are no longer m x 1 matrix, just a 1-d ndarray with m elements.)
So in Octave/Matlab, 
    ```matlab
    >> size(theta)
    >> (2, 1)
    ```
    Now, it is
    ```python
    >>> theta.shape
    >>> (2, )
    ```
* numpy.matrix is never used, just plain ol' numpy.ndarray

## Contents
#### [Exercise 1]
* Linear Regression
* Linear Regression with multiple variables
#### [Exercise 2]
* Logistic Regression
* Logistic Regression with Regularization
#### [Exercise 3]
* Multiclass Classification
* Neural Networks Prediction fuction
#### [Exercise 4]
* Neural Networks Learning
#### [Exercise 5]
* Regularized Linear Regression
* Bias vs. Variance
#### [Exercise 6](
* Support Vector Machines
* Spam email Classifier
#### [Exercise 7]
* K-means Clustering
* Principal Component Analysis
#### [Exercise 8]
* Anomaly Detection
* Recommender Systems

## Solutions
You can check out my implementation of the assignments [here](https://github.com/learning511/cs229-assignments/tree/master/Solutions). I tried to vectorize all the solutions.
