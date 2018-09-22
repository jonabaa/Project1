import numpy as np
from imageio import imread
#from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utilities import *

"""
# Load the terrain
terrain1 = imread('data_files/SRTM_data_Norway_1.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""

m, n = 100, 100
x, y = tifread(mlimit=m, nlimit=n, filename='data_files/SRTM_data_Norway_2.tif')
rbeta1 = RidgeReg(x, y, 5, 1)
plot_function_2D(5, rbeta1, m, n, 'k5RidgeLambda1')

lbeta1 = LassoReg(x, y, 5, 1)
plot_function_2D(5, lbeta1, m, n, 'k5LassoLambda1')

rbeta0 = RidgeReg(x, y, 5, 0)
plot_function_2D(5, rbeta0, m, n, 'k5OLS')
