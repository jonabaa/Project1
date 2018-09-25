import numpy as np
from imageio import imread
#from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utilities import *

"""
# Load the terrain
terrain1 = imread('data_files/SRTM_data_Norway_2.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 2')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""

m, n = 250, 250
x, y = tifread(mlimit=m, nlimit=n, filename='data_files/SRTM_data_Norway_2.tif')

plot_realdata(x, y)
obetareal = RidgeReg(x, y, 10, 0)
plot_function_2D(10, obetareal, m, n, '250x250OLSdegree10')
"""

lbetareal = LassoReg(x, y, 5, 0.01)
plot_function_2D(5, lbetareal, m, n, '250x250Lasso1e-2')
"""

"""
rbeta1 = RidgeReg(x, y, 5, 1)
plot_function_2D(5, rbeta1, m, n, 'k5RidgeLambda1')

lbeta1 = LassoReg(x, y, 5, 1)
plot_function_2D(5, lbeta1, m, n, 'k5LassoLambda1')

rbeta0 = RidgeReg(x, y, 5, 0)
plot_function_2D(5, rbeta0, m, n, 'k5OLS')

rbeta5 = RidgeReg(x, y, 5, 5)
plot_function_2D(5, rbeta5, m, n, 'k5RidgeLambda5')
"""
