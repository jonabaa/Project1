import numpy as np
from random import random, seed
from sklearn import linear_model
from imageio import imread
import scipy.stats as st


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

# Variance of beta
def VAR(x, y, y_tilde, k):
    # Need to compute
    # Var(BETA) = (X^T X)^-1 sigma^2
    # X is the construct matrix
    # sigma^2 = 1/(N - k - 1) * RSS
    m = x.shape[0]
    n = SumOneToN(k + 1)

    # allocate design matrix
    X = np.ones((m, n))


    # compute values of design matrix
    for i in range(m):
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                X[i][SumOneToN(p + 1) + j] *= x[i][0]**(p+1-j)*x[i][1]**j


    varmatrix = np.linalg.inv(((X.T).dot(X))) * ((1/(len(y)-k-1))*RSS(y, y_tilde))
    # sigma2 = ((1/(len(y)-k-1))*RSS(y, y_tilde))


    # the diagonal is the variance, other indexes represents covariances
    # only want to return the variance

    # returns the variance corresponding to each beta that was the input
    return np.diagonal(varmatrix)


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


# This function draws predictors from the square [0,1]^2 with
# uniform distribution and computes the value of the Frankefunction
# at these points. Finally it adds normally distributed noise around 0
# to the responses.
def CreateSampleData(n, s):
    x = np.random.rand(n, 2)
    y = FrankeFunction(x[:,0:1], x[:,1:2]) + s*np.random.randn(n,1)

    return x[:,0:1], x[:,1:2], y


# This just reads an mxn block of the input-file
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

    x1 = np.zeros((mlimit*nlimit, 1))
    x2 = np.zeros((mlimit*nlimit, 1))
    y = np.zeros((mlimit*nlimit, 1))

    # Seperate x1 and x2 in coloumns in x and the
    # corresponding values in y

    # Make x all combinations of the axis and y corresponds in value
    for i in range(0, mlimit):
        for j in range(0, nlimit):
            x1[i + j*mlimit] = i
            x2[i+ j*mlimit] = j

            y[i+ j*mlimit] = im[i][j]


    # x and y can be used in the regression-functions
    return x1, x2, y

def CIvar(beta, varbeta, percentile = 0.95):
    # Given a beta and variance of beta calculates
    # the confidence interval of the betas

    # stdcoeff is the z-score to the two-sided confidence interval
    stdcoeff = st.norm.ppf((1-percentile)/2)

    CIvector = np.zeros((len(beta), 2))
    for i in range(len(beta)):
        CIvector[i][0] = beta[i] + stdcoeff*np.sqrt(varbeta[i])
        CIvector[i][1] = beta[i] - stdcoeff*np.sqrt(varbeta[i])

    # CIvector returns a nx2 matrix with each row
    # representing the confidence interval to the corresponding beta
    return CIvector
