from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn import linear_model


# Residual squared ...
def RSS(y, y_tilde):
    return sum((y - y_tilde)**2)


# Mean squared error
def MSE(y, y_tilde):
    return RSS(y, y_tilde)*(1/float(y.size))


# R2-score function
def R2Score(y, y_tilde):
    return 1 - RSS(y, y_tilde)/sum((y - sum(y)/float(y.size))**2)


# Mean absolute error
def MAE(y, y_tilde):
    return sum(abs(y - y_tilde))/float(y.size)


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
    y = FrankeFunction(x[:,0], x[:,1]) + s*np.random.randn(n)

    return x, y



# Bootstrap with B resamples
def Bootstrap(n, k, s, lmb, B):
    # create sample data
    spl_x, spl_y = CreateSampleData(n, s)


    # split data in to training data and test data
    """
    train_x = spl_x[:n/2,:]
    train_y = spl_y[:n/2]

    test_x = spl_x[n/2:,:]
    test_y = spl_y[n/2:]
    """
    # compute regression coefficients
    #beta = RidgeReg(train_x, train_y, k, lmb)

    # plot
    #plot_function(k, beta)

    # Bootstrap

    """
    boots = np.zeros((train_y.size,3))
    print(boots.shape)
    print (train_x.shape)
    boots[:][0:2] = train_x
    boots[:][2] = train_y

    print(boots)
    """

    #for b in range(B):
        # draw with replacement from training data
        #boots_x = np.random.choice(train_y, len(train_y))

        # compute regression coefficients
        #beta = RidgeReg(boots_x, boots_y, k, lmb)

        # plot
        #plot_function(k, beta)


# plot the fuitted function, TO BE REMOVED
def plot_function(k, beta):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x1 = np.arange(0, 1, 0.05)
    x2 = np.arange(0, 1, 0.05)
    x1, x2 = np.meshgrid(x1,x2)

    y = Polynome(x1, x2, k, beta)

    # Plot the surface.
    surf = ax.plot_surface(x1, x2, y, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #surf2 = ax.plot_surface(x1, x2, y_f, cmap=cm.coolwarm,
    #                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# Test of LassoReg compared to RSS which is Ridge with lambda = 0
x, y = CreateSampleData(100, 1)
betaL = LassoReg(x, y, 5, 1e-10)
betaR = RidgeReg(x, y, 5, 0.0)
plot_function(5, betaL)
plot_function(5, betaR)

# Program
n = 100 # number of datapoints in sample data set
k = 5 # the order of your fit-polynomial
s = .1 # the noice coefficient
lmb = 0 # lambda determining the emphasize on minimizing beta vs cost-f.

Bootstrap(n, k, s, lmb, 10)


"""
# compute and print some statistic measures
mod_y = Polynome(spl_x[:,0], spl_x[:,1], k, beta)
print("The R2-score: %.5f." % R2Score(spl_y, mod_y))
print("The Mean squared error: %.5f." % MSE(spl_y, mod_y))
"""
"""
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""
