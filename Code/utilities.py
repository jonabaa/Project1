import numpy as np
from random import random, seed
from sklearn import linear_model
from imageio import imread


# Residual sums squared 
def RSS(y, y_tilde):
    return np.sum((y - y_tilde)**2, axis=0)


# Mean squared error
def MSE(y, y_tilde):
    return RSS(y, y_tilde)*(1/len(y))


# R2-score function
def R2Score(y, y_tilde):
    return 1 - RSS(y, y_tilde)/np.sum((y - np.sum(y, axis=0)/y.size)**2, axis=0)


# Mean absolute error
def MAE(y, y_tilde):
    return np.sum(np.abs(y - y_tilde), axis=1)/y.size


# the Franke function, f:R^2 -> R
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4


# sums up the n first numbers of N
def SumOneToN(n):
    return int((n + 1)*n/2)

# DELETE
# k-th order polynomial DELETE
def Polynome(x, y, k, beta):
    if beta.size != SumOneToN(k + 1):
        print("Function Polynom says: Bad input data.")
        return 0
    else:
        # allocate meshgrid of appropriate shape filles with constant term
        z = np.ones(x.shape)*beta[0]

        # compute function values
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                z += beta[SumOneToN(p + 1) + j]*x**(p+1-j)*y**j

        return z

# DELETE
# Fits a k-th order polynomial, p:R^2 -> R, to the given data x, y
# using Ridge regression with lambda=lmb 
def RidgeReg(x, y, k, lmb):
    print(x.shape)
    # calculate the dimensions of the design matrix
    m = x.shape[0]
    n = SumOneToN(k + 1)

    # allocate design matrix
    X = np.ones((m, n))

    # compute values of design matrix
    for i in range(m):
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                X[i][SumOneToN(p + 1) + j] *= x[i][0]**(p+1-j)*x[i][1]**j

    # compute linear regression coefficients
    beta = np.linalg.inv(X.T.dot(X) + lmb*np.identity(n)).dot(X.T).dot(y)
    
    covar_matrix = np.linalg.inv(X.T.dot(X)) #*(sum(y-sum(y)/m)/(x.shape[0] - x.shape[1] -1))
    
    return beta

# DELETE
# Fits a k-th order polynomial, p:R^2 -> R, to the given data x, y
# using Lasso regression with lambda=lmb
def LassoReg(x, y, k, lmb):
    # calculate the dimensions of the design matrix
    m = x.shape[0]
    n = SumOneToN(k + 1)

    # allocate design matrix
    X = np.ones((m, n))

    # compute values of design matrix
    for i in range(m):
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                X[i][SumOneToN(p + 1) + j] *= x[i][0]**(p+1-j)*x[i][1]**j

    #compute lasso-coefficients using scikitlearn
    lasso = linear_model.Lasso(alpha=lmb)
    lasso.fit(X, y)
    beta = lasso.coef_

    return beta


# This function draws predictors from the square [0,1]^2 with 
# uniform distribution and computes the value of the Frankefunction 
# at these points. Finally it adds normally distributed noise around 0
# to the responses.
def CreateSampleData(n, s):
    x = np.random.rand(n, 2)
    y = FrankeFunction(x[:,0:1], x[:,1:2]) + s*np.random.randn(n,1)

    return x[:,0:1], x[:,1:2], y


# Bootstrap with B resamples
#
# s = sample data
# f = Regressionfunction ((RidgeRwg or LassoReg)
# k = order of polynomial for regression function to fit
# lmb = lambda (set to 0 for OLS-regression)
# p = list of parameter estimator functions
# B = number of bootstrap-samples
#
def Bootstrap(s, f, k, lmb, p, B):
    # allocate arrays for storing bootstrap estimators
    bootstrap_estimator = np.zeros((len(p), B))

    for b in range(B):
        # draw a random bootstrap sample with replacement
        bs_s = s[np.random.choice(s.shape[0], s.shape[0]),:]

        # compute model
        x = bs_s[:,0:2]
        y = bs_s[:,2]
        beta = f(x, y, k, lmb)

        # compute y_tilde
        y_tilde = Polynome(x[:,0], x[:,1], k, beta)

        # compute estimators of parameters
        for i in range(len(p)):
            bootstrap_estimator[i,b] = p[i](y, y_tilde)

    # compute bootstrap mean of estimators
    estimator_mean = np.sum(bootstrap_estimator, axis=1)/B

    return estimator_mean


# Bootstrap with B resamples
#
# s = sample data
# f = Regressionfunction ((RidgeRwg or LassoReg)
# k = order of polynomial for regression function to fit
# lmb = lambda (set to 0 for OLS-regression)
# B = number of bootstrap-samples
#
def Bootstrap2(s, f, k, lmb, B):
    # allocate arrays for storing bootstrap estimators
    bootstrap_estimator = np.zeros((2, B))

    for b in range(B):
        # draw a random bootstrap sample with replacement
        bs_s = s[np.random.choice(s.shape[0], s.shape[0]),:]

        # compute model
        x = bs_s[:,0:2]
        y = bs_s[:,2]
        beta = f(x, y, k, lmb)

        # compute y_tilde
        y_tilde = Polynome(x[:,0:1], x[:,1:2], k, beta)
        if b == 0:
            y_tilde_matrix = y_tilde
        else:
            y_tilde_matrix = np.concatenate([y_tilde_matrix, y_tilde], axis=1)
        # compute estimators of parameters
        bootstrap_estimator[0,b] = MSE(y, y_tilde_matrix[:,b])
        bootstrap_estimator[1,b] = R2Score(y, y_tilde_matrix[:,b])
    """
    # compute variance
    for b in range(B):
        y_tilde_matrix[:,b] = (y_tilde_matrix[:,b] - np.sum(y_tilde_matrix, axis=1)/B)**2
    var = np.sum(y_tilde_matrix, axis=1)/B
    var = sum(var)
    print(var)
    """
    #compute expected value in x over the bootstrapsamples
    E_L = (np.sum(y_tilde_matrix, axis=1)/B).reshape((len(y_tilde),1))

    # compute variance
    var = sum(np.sum((y_tilde_matrix - E_L)**2, axis=1)/B)
    
    # compute bias
    bias = np.sum((y - E_L)**2)
    
    # compute bootstrap mean of estimators
    estimator_mean = np.sum(bootstrap_estimator, axis=1)/B

    # do some printing for test purposes
    print("VAR: %.4f." % var)
    print("BIAS: %.4f." % bias)
    print("Bootstrap mean of MSE: %.4f." % estimator_mean[0])
    print("Bootstrap mean of r2Score: %.4f." % estimator_mean[1])

    return_values = np.zeros((4,1))
    return_values[0] = bias
    return_values[1] = var
    return_values[2] = estimator_mean[0]
    return_values[3] = estimator_mean[1]

    return return_values


def tifread(mlimit=100, nlimit=100, filename='data_files/SRTM_data_Norway_1.tif',):
    # Sets default to SRTM data Norway 1
    im = imread(filename)
    m, n = im.shape
    if m < mlimit:
        print("Decrease mlimit")
        return None, None
    if n < nlimit:
        print("Decrease nlimit")
        return None, None

    x = np.zeros((mlimit*nlimit, 2))
    y = np.zeros((mlimit*nlimit, 1))

    # Seperate x1 and x2 in coloumns in x and the
    # corresponding values in y

    # Make x all combinations of the axis and y corresponds in value
    for i in range(0, mlimit):
        for j in range(0, nlimit):
            x[i + j*mlimit][0] = i
            x[i+ j*mlimit][1] = j

            y[i+ j*mlimit] = im[i][j]


    # x and y can be used in the regression-functions
    return x, y


