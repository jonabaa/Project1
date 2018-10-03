from RidgeLinearModel import RidgeLinearModel
from LassoLinearModel import LassoLinearModel
from OLSLinearModel import OLSLinearModel
from utilities import *
from visualization import *
from resampling import *
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(1254)
x1, x2, y = CreateSampleData(100, .0)
lmb = .0
B = 50
K = 7

"""
model = [1,2,3,4,5,6]

for i in range(6):
    model[i] = OLSLinearModel(i)
    model[i].fit(x1, x2, y)
    model[i].predict(x1, x2)
    print("OLS model: lambda=%.3f, degree=%d." % (lmb, i))
    print("RSS=%4f, MSE=%.6f, R2Score=%.4f" % (model[i].get_RSS(), model[i].get_MSE(), model[i].get_R2Score()))
    print("Variances of coefficients:")
    print(model[i].get_variance_of_coefficients())
    print("")
"""
returnvalues = np.zeros((4,K))

for k in range(K):
    print("Lambda=0, k=%d" % k)
    returnvalues[:,k:k+1] = BootstrapRidge(x1, x2, y, k, lmb, B)
    print("")

x = np.linspace(0,K,K)
plt.plot(x, np.sqrt(returnvalues[0,:]))
plt.plot(x, returnvalues[1,:])
plt.xlabel("Degree of polynomial")
plt.legend(["bias","varians"])
plt.show()

# IMPLEMENTATION OF LASSO   
# function to minimize y - y_tilde
# y - X*beta
# we are to optimize a linear function over a polyhedron
# hence we can use the simplex method and theory from linear optimization
# to find the minimum of this function. Because it is a a convex function
# on a closed convex set it must have a minimum. I guess we also can 
# prove that this minimum is unique due to the shape of the set we
# are minimizing over.
