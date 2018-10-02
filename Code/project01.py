from utilities import *



# PROJECT 01 SOLUTIONS

# Part a)-------------------------
# Ordinary Least Square on the Franke function
# with resampling

# This is just Ridge with lambda = 0
# Need to find variance of beta, compute MSE
k = 5
x, y = CreateSampleData(1000, 0.01)
obeta0 = RidgeReg(x, y, k, 0)

y_tilde = y_predict(x, k, obeta0)
varvector = VAR(x, y, y_tilde, k)
CI = CIvar(obeta0, varvector, percentile = 0.95)
print(CI)

# Check values of this with bootstrap
s = np.c_[x, y]
boots = Bootstrap2(s, RidgeReg, k, 0, 10)
#
# Plots different scores with MSE and R2
#plotscores(RidgeReg, s,'Ridge' ,lambdasteps=10)

#plot_function_2D(5, obeta0, 1, 1, 'a-OLS')


# Part b)-------------------------
# Ridge Regression on the Franke function
# with resampling

rbeta1 = RidgeReg(x, y, 5, 0.1)
#plotscores(RidgeReg, s,'Ridge' ,lambdasteps=10, karray=[2, 5, 10],savefig=False)
# Check values of this with bootstrap


#plot_function_2D(5, rbeta1, 1, 1, 'b-RidgeLambda01')

# Part c)-------------------------
# Lasso Regression on the Franke function
# with resampling

lbeta1 = LassoReg(x, y, 5, 0.01)
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
x, y = tifread(mlimit=m, nlimit=n, filename='data_files/SRTM_data_Norway_2.tif')
"""
#plot_realdata(x, y, '100x100nor2')

# Potentially download own data from website
# Just get the data represented in variebles
# and maybe split it up if it's too much


# Part e)-------------------------
# OLS, Ridge and Lasso regression with resampling


#obetareal = RidgeReg(x, y, 5, 0)
#plot_function_2D(5, obetareal, m, n, 'e-OLS')
#s = np.c_[x, y]
#plotscores(RidgeReg, s,'RidgeReal' ,lambdasteps=5)

#rbetareal = RidgeReg(x, y, 5, 0.1)
#plot_function_2D(5, rbetareal, m, n, 'e-Ridge01')

#lbetareal = LassoReg(x, y, 5, 0.01)
#plot_function_2D(5, lbetareal, m, n, 'e-Lassolamda001')

# Basicly repeat of a-c just with real data
