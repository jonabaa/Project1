from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import arange, meshgrid, concatenate

# plot the fitted function, TO BE REMOVED
#
# k = order of polynome
# beta = coefficients of polynome in ascending order
#
def plot_function_3D(k, beta, m, n):
    # Plots the figure in 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x1 = arange(0, m, 0.1)
    x2 = arange(0, n, 0.1)
    x1, x2 = meshgrid(x1,x2)

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


# Draws a 3D-plot of the given model
def plot_model_3D(model):
    # make instance of plot object
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # set up grid of independent variables
    x1 = arange(0, 1, 0.01)
    x2 = arange(0, 1, 0.01)
    x1, x2 = meshgrid(x1,x2)

    # let model predict dependent variables
    y = model.predict(x1, x2)

    # plot the surface
    surf = ax.plot_surface(x1, x2, y, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # customize the z axis
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


# Takes degree of function, the beta and the scale of plotting as input
def plot_function_2D(k, beta, m, n, navn, savefig=False):
    # Plots the figure in 2D
    x1 = arange(0, m, 0.05)
    x2 = arange(0, n, 0.05)

    x1, x2 = meshgrid(x1, x2)

    y = Polynome(x1, x2, k, beta)

    fig = plt.figure()
    plt.pcolormesh(x1, x2 ,y , cmap='inferno')
    plt.colorbar()
    plt.title('Plot of model')
    plt.xlabel('X')
    plt.ylabel('Y')
    if savefig:
        fig.savefig('figs/%s.png'%(navn), dpi=fig.dpi)
    plt.show()


# This function creates and stores two plots:
# The first one plots Bias and Variance against order of the polynome/model
# The second one plots MSE and R2Score against order of the polynome/model
#
# @RegMethod: regressionmethod to be used when creating models
# @K: fit polynomials of degree 0 up to K
# @lmb: the lambda to use if RegMethod is Lasso or Ridge
# @B: number of bootstrapsamples
#
def generate_errorplots(RegMethod, K, lmb, B=100):
    lmb = .1 # lambda
    K = 10 # compute for all degress up to K

    x, y = CreateSampleData(500, .1)
    s = concatenate([x,y], axis=1)

    for k in range(K):
        if k == 0:
            return_values = Bootstrap2(s, RegMethod, k, lmb, B)
        else:
            return_values = np.concatenate([return_values, Bootstrap2(s, RidgeReg, k, lmb, B)], axis=1)

    x = range(K)

    # save the plots to files
    filename1 = "OLS-test1"
    filename2 = "OLS-test2"

    plt.plot(x, return_values[0,:]/1000, label="Bias^2/1000")
    plt.plot(x, return_values[1,:], label="Var")
    plt.legend()
    plt.savefig(filename1)

    plt.gcf().clear()

    plt.plot(x, return_values[2,:], label="MSE")
    plt.plot(x, return_values[3,:], label="R2Score")
    plt.legend()
    plt.savefig(filename2)

def plotscores(RegMethod, plotname , karray=[3,4,5], lambdasteps=5, savefig=False):

    lmbx = np.logspace(-2, 4, lambdasteps)
    r2scores = np.zeros((len(karray), len(lmbx)))
    msescores = np.zeros((len(karray),len(lmbx)))

    # Definer x1, x2, y, k, lmb og B til bootstrap


    for j in range(len(karray)):
        for i in range(len(lmbx)):
            # Will implement for function for each k
            bias, var, mse, r2 = BootstrapRidge(s, function, karray[j], lmbx[i], 10)
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
