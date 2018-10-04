from utilities import *
from OLSLinearModel import OLSLinearModel
from LassoLinearModel import LassoLinearModel
from resampling import *
from visualization import *



# PROJECT 01 SOLUTIONS

# Part a)-------------------------
# Ordinary Least Square on the Franke function
# with resampling

# This is just Ridge with lambda = 0
# Need to find variance of beta, compute MSE
import pandas as pd
klist = []
CI = []
MSE = []
R2 = []


for k in range(1,6):
    x1, x2, y = CreateSampleData(1000, 0.01)
    OLSmodel = OLSLinearModel(k)
    OLSmodel.fit(x1, x2, y)

    OLS_var = OLSmodel.get_sample_variance_of_betas()
    print(OLS_var)
    OLS_CI = OLSmodel.get_CI_of_beta()
    OLS_MSE = OLSmodel.get_MSE(x1, x2, y)
    OLS_R2 = OLSmodel.get_R2Score(x1, x2, y)

    klist.append(k)
    MSE.append(OLS_MSE)
    R2.append(OLS_R2)
    CI.append(OLS_CI)

d ={'R2' : pd.Series(R2, index=klist),
   'MSE' : pd.Series(MSE, index=klist),
    'CI' : pd.Series(CI, index=klist)}

table = pd.DataFrame(d)
#print(table)



# Check values of this with bootstrap
#OLSboots = BootstrapRidge(x1, x2, y, k, 0, 100)
#
# Plots different scores with MSE and R2
#plotscores(RidgeReg, s,'Ridge' ,lambdasteps=10)

#plot_function_2D(5, obeta0, 1, 1, 'a-OLS')


# Part b)-------------------------
# Ridge Regression on the Franke function
# with resampling

RidgeModel = RidgeLinearModel(lmb=0.01, k=k)
RidgeModel.fit(x1, x2, y)
#print(RidgeModel.beta)
#plotscores(RidgeReg, s,'Ridge' ,lambdasteps=10, karray=[2, 5, 10],savefig=False)
# Check values of this with bootstrap


#plot_function_2D(5, rbeta1, 1, 1, 'b-RidgeLambda01')

# Part c)-------------------------
# Lasso Regression on the Franke function
# with resampling
"""
"""
LassoModel = LassoLinearModel(lmb=0.01, k=k)
LassoModel.fit(x1, x2, y)
Lasso_VAR = LassoModel.get_variance_of_betas(100)
Lasso_CI = LassoModel.get_CI_of_beta()
# Check values of this with bootstrap
# Plots different scores with MSE and R2
#plotscores(LassoReg, s, 'Lasso',lambdasteps=10)
#plot_function_2D(5, lbeta1, 1, 1, 'c-LassoLambda001')

# Discuss which method is best
# Compare the bootstrap-values etc.


# Part d)-------------------------
# Introducing real data

# This splits the data into a chunk 100x100 up in the right corner

"""
m, n = 100, 100
x1, x2, y = tifread(mlimit=m, nlimit=n, filename='data_files/SRTM_data_Norway_2.tif')

#plot_realdata(x, y, '100x100nor2')

# Potentially download own data from website
# Just get the data represented in variebles
# and maybe split it up if it's too much


# Part e)-------------------------
# OLS, Ridge and Lasso regression with resampling


RealOLSModel = OLSLinearModel(k)
#plot_function_2D(5, obetareal, m, n, 'e-OLS')
#s = np.c_[x, y]
#plotscores(RidgeReg, s,'RidgeReal' ,lambdasteps=5)

#rbetareal = RidgeReg(x, y, 5, 0.1)
#plot_function_2D(5, rbetareal, m, n, 'e-Ridge01')
RealLassoModel = LassoLinearModel(lmb=0.01, k=k)
RealLassoModel.fit(x1, x2, y)
print(RealLassoModel.beta)
print(RealLassoModel.get_variance_of_coefficients())
print(RealLassoModel.get_R2Score())
"""
#lbetareal = LassoReg(x, y, 5, 0.01)
#plot_function_2D(5, lbetareal, m, n, 'e-Lassolamda001')

# Basicly repeat of a-c just with real data
