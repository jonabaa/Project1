from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn import linear_model
from imageio import imread
import scipy.stats as st


# Residual squared ...
def RSS(y, y_tilde):
    return sum((y - y_tilde)**2)


# Mean squared error
def MSE(y, y_tilde):
    return RSS(y, y_tilde)*(1/len(y))


# R2-score function
def R2Score(y, y_tilde):
    return 1 - RSS(y, y_tilde)/sum((y - sum(y)/float(y.size))**2)


# Mean absolute error
def MAE(y, y_tilde):
    return sum(abs(y - y_tilde))/float(y.size)

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


# k-th order polynomial
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


# Fits a k-th order polynomial, p:R^2 -> R, to the given data x, y
# using Ridge regression
def RidgeReg(x, y, k, lmb):
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

    return beta


# Fits a k-th order polynomial, p:R^2 -> R, to the given data x, y
# using Ridge regression
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


# create n sample data points
def CreateSampleData(n, s):
    x = np.random.rand(n, 2)
    y = FrankeFunction(x[:,0:1], x[:,1:2])# + s*np.random.randn((n,1))

    return x, y


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
    #bootstrap_estimator_mean = np.zeros((len(p),))

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

    print("VAR: %f.2." % var)
    print("BIAS: %f.2." % bias)
    print("Bootstrap mean of MSE: %f.2." % estimator_mean[0])
    print("Bootstrap mean of r2Score: %f.2." % estimator_mean[1])

    return estimator_mean

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



# plot the fitted function, TO BE REMOVED
#
# k = order of polynome
# beta = coefficients of polynome in ascending order
#
def plot_function_3D(k, beta, m, n):
    # Plots the figure in 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x1 = np.arange(0, m, 0.1)
    x2 = np.arange(0, n, 0.1)
    x1, x2 = np.meshgrid(x1,x2)

    y = Polynome(x1, x2, k, beta)

    # Plot the surface.
    surf = ax.plot_surface(x1, x2, y, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #surf2 = ax.plot_surface(x1, x2, y_f, cmap=cm.coolwarm,
    #                       linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)                   #Tar bort limits, skal teste tif-filen
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_function_3D(k, beta, m, n):
    # Plots the figure in 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x1 = np.arange(0, m, 0.1)
    x2 = np.arange(0, n, 0.1)
    x1, x2 = np.meshgrid(x1,x2)

    y = Polynome(x1, x2, k, beta)

    # Plot the surface.
    surf = ax.plot_surface(x1, x2, y, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #surf2 = ax.plot_surface(x1, x2, y_f, cmap=cm.coolwarm,
    #                       linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)                   #Tar bort limits, skal teste tif-filen
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_function_2D(k, beta, m, n, navn, savefig=False):
    # Plots the figure in 2D
    x1 = np.arange(0, m, 0.05)
    x2 = np.arange(0, n, 0.05)

    x2, x1 = np.meshgrid(x1, x2)

    y = Polynome(x1, x2, k, beta).T
    y1 = np.flip(y, axis=1)

    fig = plt.figure()
    plt.pcolormesh(x1, x2 ,y1, cmap='inferno')
    plt.colorbar()
    plt.title(navn)
    plt.xlabel('X')
    plt.ylabel('Y')
    if savefig:
        fig.savefig('figs/%s.png'%(navn), dpi=fig.dpi)
    plt.show()

# Plots real data to compare
def plot_realdata(x, y, navn, savefig=False):
    x2 = x[:, 1].reshape((len(x[:,1]), 1))
    x1 = x[:, 0].reshape((len(x[:,0]), 1))

    N = int(len(y)**0.5)
    y1 = y.reshape(N, N).T
    # Transpose the y1 to fit the axis properly

    fig = plt.figure()
    plt.imshow(y1, extent=(np.amin(x1), np.amax(x1), np.amin(x2), np.amax(x2)), cmap='inferno')
    plt.colorbar()
    plt.title(navn)
    plt.xlabel('X')
    plt.ylabel('Y')
    if savefig:
        fig.savefig('figs/%s.png'%(navn), dpi=fig.dpi)
    plt.show()

# Make a function which plots different scores of R2 and MSE
# First is with same model, but different k-s and lambda
# Plot R2/MSE by the value of lambda. PLot for different degrees

def plotscores(function, s, plotname , karray=[3,4,5], lambdasteps=5, savefig=False):

    lmbx = np.logspace(-2, 4, lambdasteps)
    r2scores = np.zeros((len(karray), len(lmbx)))
    msescores = np.zeros((len(karray),len(lmbx)))

    for j in range(len(karray)):
        for i in range(len(lmbx)):
            # Will implement for function for each k
            mse, r2 = Bootstrap2(s, function, karray[j], lmbx[i], 10)
            r2scores[j][i] = r2
            msescores[j][i] = mse

    fig = plt.figure()

    for i in range(len(karray)):
        plt.plot(lmbx,r2scores[i], label='degree= %s'%karray[i])
    plt.legend()
    plt.title('R2 of %s' %plotname)
    plt.xlabel('lambda')
    plt.xscale('log')
    plt.ylabel('R2')
    if savefig:
        fig.savefig('scorefigs/R2%s.png'%(plotname), dpi=fig.dpi)
    plt.show()

    for i in range(len(karray)):
        plt.plot(lmbx,msescores[i], label='degree= %s'%karray[i])
    plt.legend()
    plt.title('MSE of %s' %plotname)
    plt.xlabel('lambda')
    plt.xscale('log')
    plt.ylabel('MSE')
    if savefig:
        fig.savefig('scorefigs/MSE%s.png'%(plotname), dpi=fig.dpi)
    plt.show()

def y_predict(x, k, beta):
    # Given x, k and beta, returns predicted y
    y_tilde = np.ones((x.shape[0], 1))*beta[0]

    for i in range(x.shape[0]):
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                y_tilde[i] += beta[SumOneToN(p + 1) + j]*x[i][0]**(p+1-j)*x[i][1]**j

    return y_tilde

def CIvar(beta, varbeta, percentile = 0.95):
    # Given a beta and variance of beta calculates
    # the confidence interval of the betas

    stdcoeff = st.norm.ppf((1-percentile)/2)
    print(stdcoeff)
    CIvector = np.zeros((len(beta), 2))
    for i in range(len(beta)):
        CIvector[i][0] = beta[i] + stdcoeff*np.sqrt(varbeta[i])
        CIvector[i][1] = beta[i] - stdcoeff*np.sqrt(varbeta[i])

    return CIvector
