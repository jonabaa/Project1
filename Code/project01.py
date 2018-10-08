from utilities import *
from OLSLinearModel import OLSLinearModel
from LassoLinearModel import LassoLinearModel
from resampling import *
from visualization import *
import pandas as pd

#----------------------------------------#
# ALL OF THIS DATA IS STORED IN DATA.TXT #
# IF YOU WANT TO CHECK IF THE CODE WORKS #
# RUN PARTS OF IT, AS THE WHOLE THING    #
# TAKES A while                          #
#----------------------------------------#


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

        MSE = model.get_MSE(x1, x2, y)
        R2 = model.get_R2Score(x1, x2, y)

        klist.append(k)
        MSElist.append(MSE)
        R2list.append(R2)

    d ={'R2' : pd.Series(R2list, index=klist),
        'MSE' : pd.Series(MSElist, index=klist)}

    return pd.DataFrame(d)

# PROJECT 01 SOLUTIONS
x1, x2, y = CreateSampleData(1000, 0.01)
"""
# Part a)-------------------------
# Ordinary Least Square on the Franke function
# with resampling

# This is just Ridge with lambda = 0
# Need to find variance of beta, compute MSE
print('OLS Test Data')
print(MSEandR2table(x1,x2,y,OLSLinearModel, 5))
print()

OLSmodel = OLSLinearModel(5)
OLSmodel.fit(x1, x2, y)

OLS_var = OLSmodel.get_variance_of_betas()
OLS_CI = OLSmodel.get_CI_of_beta()

print('Var of Beta')
print(OLS_var)
print()

print('95-percentage CI of betas')
print(OLS_CI)
print()

for i in range(1, 6):
    print('Bootstrap-values from degree of %s and 100 bootstrap-samples'%i)
    OLSboots = BootstrapRidge(x1, x2, y, i, 0, 100)
    print()
#
# Plots different scores with MSE and R2
#plotscores(RidgeReg, s,'Ridge' ,lambdasteps=10)

#plot_function_2D(5, obeta0, 1, 1, 'a-OLS')


# Part b)-------------------------
# Ridge Regression on the Franke function
# with resampling
print('Ridge Test Data lmb = 0.1')
print(MSEandR2table(x1,x2,y,RidgeLinearModel, 5))
print()
Ridgemodel = RidgeLinearModel(0.1, 5)
Ridgemodel.fit(x1, x2, y)

Ridge_var = Ridgemodel.get_variance_of_betas()
Ridge_CI = Ridgemodel.get_CI_of_beta()

print('Var of Beta')
print(Ridge_var)

print('95-percentage CI of betas')
print(Ridge_CI)
#plotscores(RidgeReg, s,'Ridge' ,lambdasteps=10, karray=[2, 5, 10],savefig=False)
# Check values of this with bootstrap

for i in range(1, 6):
    for j in [0.1, 1, 10]:
        print('Bootstrap-values from degree of %s, lmb = %s and 100 bootstrap-samples'%(i, j))
        OLSboots = BootstrapRidge(x1, x2, y, i, j, 100)
        print()

"""

#plot_function_2D(5, rbeta1, 1, 1, 'b-RidgeLambda01')

# Part c)-------------------------
# Lasso Regression on the Franke function
# with resampling

"""
print('Lasso Test Data lmb = 0.1')
print(MSEandR2table(x1,x2,y,LassoLinearModel, 5))
print()

Lassomodel = LassoLinearModel(0.01, 5)
Lassomodel.fit(x1, x2, y)

Lasso_var = Lassomodel.get_variance_of_betas()
Lasso_CI = Lassomodel.get_CI_of_beta()

print('Var of Beta')
print(Lasso_var)

print('95-percentage CI of betas')
print(Lasso_CI)

# Check values of this with bootstrap

# FIX THINGS INSIDE??
for i in range(1, 6):
    for j in [0.1, 1, 10]:
        print('Bootstrap-values from degree of %s, lmb = %s and 100 bootstrap-samples'%(i, j))
        OLSboots = BootstrapLasso(x1, x2, y, i, j, 100)
        print()
# Plots different scores with MSE and R2
#plotscores(LassoReg, s, 'Lasso',lambdasteps=10)
#plot_function_2D(5, lbeta1, 1, 1, 'c-LassoLambda001')

# Discuss which method is best
# Compare the bootstrap-values etc.


# Part d)-------------------------
# Introducing real data

# This splits the data into a chunk 100x100 up in the right corner

"""
#----------------------#
#                      #
#     REAL DATA        #
#                      #
#----------------------#

# Decide how much of the chunk we analyze

m, n = 100, 100
x1, x2, y = tifread(mlimit=m, nlimit=n, filename='data_files/SRTM_data_Norway_2.tif')
#plot_realdata(x, y, '100x100nor2')

# Potentially download own data from website
# Just get the data represented in variebles
# and maybe split it up if it's too much


# Part e)-------------------------
# OLS, Ridge and Lasso regression with resampling

#--OLS
"""

print('OLS Real Data')
print(MSEandR2table(x1,x2,y,OLSLinearModel, 5))
print()

OLSmodel = OLSLinearModel(5)
OLSmodel.fit(x1, x2, y)

OLS_var = OLSmodel.get_variance_of_betas()
OLS_CI = OLSmodel.get_CI_of_beta()

print('Var of Beta')
print(OLS_var)
print()

print('95-percentage CI of betas')
print(OLS_CI)
print()
# Need to fix bootstrap for real values
for i in range(1, 6):
    print('Bootstrap-values from degree of %s and 100 bootstrap-samples'%i)
    OLSboots = BootstrapRidge(x1, x2, y, i, 0, 100)
    print()
#plot_function_2D(5, obetareal, m, n, 'e-OLS')
"""
"""
"""
#s = np.c_[x, y]
#plotscores(RidgeReg, s,'RidgeReal' ,lambdasteps=5)

"""
#--Ridge--
print('Ridge Real Data lmb = 0.1')
print(MSEandR2table(x1,x2,y,RidgeLinearModel, 5))
print()
Ridgemodel = RidgeLinearModel(0.1, 5)
Ridgemodel.fit(x1, x2, y)

Ridge_var = Ridgemodel.get_variance_of_betas()
Ridge_CI = Ridgemodel.get_CI_of_beta()

print('Var of Beta')
print(Ridge_var)

print('95-percentage CI of betas')
print(Ridge_CI)

#plot_function_2D(5, rbetareal, m, n, 'e-Ridge01')

#Bootstrap
for i in range(1, 6):
    for j in [0.1]:
        print('Bootstrap-values from degree of %s, lmb = %s and 100 bootstrap-samples'%(i, j))
        BootstrapRidge(x1, x2, y, i, j, 100)
        print()
"""

"""
#--Lasso--
print('Lasso Real Data lmb = 0.1')
print(MSEandR2table(x1,x2,y,LassoLinearModel, 5))
print()

Lassomodel = LassoLinearModel(0.01, 5)
Lassomodel.fit(x1, x2, y)

Lasso_var = Lassomodel.get_variance_of_betas()
Lasso_CI = Lassomodel.get_CI_of_beta()

print('Var of Beta')
print(Lasso_var)

print('95-percentage CI of betas')
print(Lasso_CI)

#Bootstrap
for i in range(1, 6):
    for j in [0.1, 1, 10]:
        print('Bootstrap-values from degree of %s, lmb = %s and 100 bootstrap-samples'%(i, j))
        BootstrapLasso(x1, x2, y, i, j, 100)
        print()

#lbetareal = LassoReg(x, y, 5, 0.01)
#plot_function_2D(5, lbetareal, m, n, 'e-Lassolamda001')

# Basicly repeat of a-c just with real data
"""

# Conclusion

# Plot R2-score of real data
klist = [1,2,3,4,5]
RidgeR2 = []
LassoR2 = []
RidgeMSE = []
LassoMSE = []


for i in klist:
    for j in [0.1]:
        RidgeR2.append(BootstrapRidge(x1, x2, y, i, j, 40)[3])
        LassoR2.append(BootstrapLasso(x1,x2,y,i,j,40)[3])
        RidgeMSE.append(BootstrapRidge(x1, x2, y, i, j, 40)[2])
        LassoMSE.append(BootstrapLasso(x1,x2,y,i,j,40)[2])

plt.plot(klist, RidgeR2)
plt.plot(klist, LassoR2)
plt.ylabel('R2-score')
plt.xlabel('degree')
plt.legend(('Ridge', 'Lasso'))
plt.show()
plt.plot(klist,RidgeMSE)
plt.plot(klist, LassoMSE)
plt.ylabel('MSE')
plt.xlabel('degree')
plt.legend(('Ridge', 'Lasso'))
plt.show()
