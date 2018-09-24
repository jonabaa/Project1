from utilities import *



# PROJECT 01 SOLUTIONS

# Part a)-------------------------
# Ordinary Least Square on the Franke function
# with resampling

# This is just Ridge with lambda = 0
# Need to find variance of beta, compute MSE

x, y = CreateSampleData(100, 0.01)
obeta0 = RidgeReg(x, y, 5, 0)
# Check values of this with bootstrap
plot_function_2D(5, obeta0, 1, 1, 'partaOLSlambda1')


# Part b)-------------------------
# Ridge Regression on the Franke function
# with resampling

rbeta1 = RidgeReg(x, y, 5, 0.01)
# Check values of this with bootstrap
plot_function_2D(5, rbeta1, 1, 1, 'partbRidgeLambda001')

# Part c)-------------------------
# Lasso Regression on the Franke function
# with resampling

lbeta1 = LassoReg(x, y, 5, 0.01)
# Check values of this with bootstrap
plot_function_2D(5, lbeta1, 1, 1, 'partcLassoLambda001')

# Discuss which method is best
# Compare the bootstrap-values etc.


# Part d)-------------------------
# Introducing real data

# Potentially download own data from website
# Just get the data represented in variebles
# and maybe split it up if it's too much

# This splits the data into a chunk 100x100 up in the right corner
m, n = 100, 100
x, y = tifread(mlimit=m, nlimit=n, filename='data_files/SRTM_data_Norway_2.tif')


# Part e)-------------------------
# OLS, Ridge and Lasso regression with resampling

obetareal = RidgeReg(x, y, 5, 0)
plot_function_2D(5, obetareal, m, n, 'parteOLS')

rbetareal = RidgeReg(x, y, 5, 0.01)
plot_function_2D(5, rbetareal, m, n, 'parteRidge')

lbetareal = LassoReg(x, y, 5, 0.01)
plot_function_2D(5, lbetareal, m, n, 'parteLasso')


# Basicly repeat of a-c just with real data
