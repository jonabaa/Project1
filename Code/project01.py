from utilities import *
from OLSLinearModel import OLSLinearModel
from LassoLinearModel import LassoLinearModel
from resampling import *
from visualization import *
import pandas as pd



# PROJECT 01 SOLUTIONS

# Part a)-------------------------
# Ordinary Least Square on the Franke function
# with resampling

# This is just Ridge with lambda = 0
# Need to find variance of beta, compute MSE
def MSEandR2table(x1, x2, y, RegMethod, krange, lmb=0.1):
    klist = []
    MSElist = []
    R2list = []
    for k in range(1, krange+1):
        if RegMethod is OLSLinearModel:
            model = RegMethod(k)
        else:
            model = RegMethod(lmb, k)
        model.fit(x1, x2, y)

        var = model.get_variance_of_betas()
        MSE = model.get_MSE(x1, x2, y)
        R2 = model.get_R2Score(x1, x2, y)

        klist.append(k)
        MSElist.append(MSE)
        R2list.append(R2)

    d ={'R2' : pd.Series(R2list, index=klist),
        'MSE' : pd.Series(MSElist, index=klist)}

    return pd.DataFrame(d)







x1, x2, y = CreateSampleData(1000, 0.01)
"""
print('OLS Test Data')
print(MSEandR2table(x1,x2,y,OLSLinearModel, 5))
print()
"""


# Check values of this with bootstrap
#OLSboots = BootstrapRidge(x1, x2, y, k, 0, 100)
#
# Plots different scores with MSE and R2
#plotscores(RidgeReg, s,'Ridge' ,lambdasteps=10)

#plot_function_2D(5, obeta0, 1, 1, 'a-OLS')


# Part b)-------------------------
# Ridge Regression on the Franke function
# with resampling
"""
print('Ridge Test Data')
print(MSEandR2table(x1,x2,y,RidgeLinearModel, 5))
print()
"""
#print(RidgeModel.beta)
#plotscores(RidgeReg, s,'Ridge' ,lambdasteps=10, karray=[2, 5, 10],savefig=False)
# Check values of this with bootstrap


#plot_function_2D(5, rbeta1, 1, 1, 'b-RidgeLambda01')

# Part c)-------------------------
# Lasso Regression on the Franke function
# with resampling
"""
print('Lasso Test Data')
print(MSEandR2table(x1,x2,y,LassoLinearModel, 5))
"""
# Check values of this with bootstrap
# Plots different scores with MSE and R2
#plotscores(LassoReg, s, 'Lasso',lambdasteps=10)
#plot_function_2D(5, lbeta1, 1, 1, 'c-LassoLambda001')

# Discuss which method is best
# Compare the bootstrap-values etc.


# Part d)-------------------------
# Introducing real data

# This splits the data into a chunk 100x100 up in the right corner

m, n = 100, 100
x1, x2, y = tifread(mlimit=m, nlimit=n, filename='data_files/SRTM_data_Norway_2.tif')

print('Lasso Real Data')
print(MSEandR2table(x1,x2,y,LassoLinearModel, 5))
"""
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
#lbetareal = LassoReg(x, y, 5, 0.01)
#plot_function_2D(5, lbetareal, m, n, 'e-Lassolamda001')

# Basicly repeat of a-c just with real data
"""
