from RidgeLinearModel import RidgeLinearModel
from LassoLinearModel import LassoLinearModel
import numpy as np
from utilities import FrankeFunction

# Bootstrap with B resamples
#
# x1: first predictor
# x2: second predictor
# y: responses
# k = order of polynomial for regression function to fit
# lmb = lambda (set to 0 for OLS-regression)
# B = number of bootstrap-samples
#
def BootstrapRidge(x1, x2, y, k, lmb, B):
    # allocate arrays for storing bootstrap estimators
    bootstrap_estimator = np.zeros((2, B))

    for b in range(B):
        # draw a random bootstrap sample with replacement
        c = np.random.choice(len(x1), len(x1))
        bs_x1 = x1[c]
        bs_x2 = x2[c]
        bs_y = y[c]

        # fit model to bootstrap sample
        model = RidgeLinearModel(lmb, k)
        model.fit(bs_x1, bs_x2, bs_y)

        # compute and store y_tilde (prediction of model given x1, x2)
        y_tilde = model.predict(x1, x2)

        if b == 0:
            y_tilde_matrix = y_tilde
        else:
            y_tilde_matrix = np.concatenate([y_tilde_matrix, y_tilde], axis=1)
        # compute estimators of parameters
        bootstrap_estimator[0,b] = model.get_MSE(x1, x2, y)
        bootstrap_estimator[1,b] = model.get_R2Score(x1, x2, y)

    #compute expected value in each x over the bootstrapsamples
    E_L = (np.mean(y_tilde_matrix, axis=1, keepdims=True))

    # compute bias
    bias = np.mean((y - E_L)**2)

    # compute variance
    var = np.mean(np.mean((y_tilde_matrix - E_L)**2, axis=1, keepdims=True))

    # compute bootstrap mean of estimators
    estimator_mean = np.sum(bootstrap_estimator, axis=1)/B

    # do some printing for test purposes
    print("VAR: %f" % var)
    print("BIAS: %f" % bias)
    print("Bootstrap mean of MSE: %.4f" % estimator_mean[0])
    print("Bootstrap mean of r2Score: %.4f" % estimator_mean[1])

    return_values = np.zeros((4,1))
    return_values[0] = bias
    return_values[1] = var
    return_values[2] = estimator_mean[0]
    return_values[3] = estimator_mean[1]

    return return_values

def BootstrapLasso(x1, x2, y, k, lmb, B):
    # allocate arrays for storing bootstrap estimators
    bootstrap_estimator = np.zeros((2, B))

    for b in range(B):
        # draw a random bootstrap sample with replacement
        c = np.random.choice(len(x1), len(x1))
        bs_x1 = x1[c]
        bs_x2 = x2[c]
        bs_y = y[c]

        # fit model to bootstrap sample
        model = LassoLinearModel(lmb, k)
        model.fit(bs_x1, bs_x2, bs_y)

        # compute and store y_tilde (prediction of model given x1, x2)
        y_tilde = model.predict(x1, x2)

        if b == 0:
            y_tilde_matrix = y_tilde
        else:
            y_tilde_matrix = np.concatenate([y_tilde_matrix, y_tilde], axis=1)
        # compute estimators of parameters
        bootstrap_estimator[0,b] = model.get_MSE(x1, x2, y)
        bootstrap_estimator[1,b] = model.get_R2Score(x1, x2, y)

    #compute expected value in each x over the bootstrapsamples
    E_L = (np.mean(y_tilde_matrix, axis=1, keepdims=True))

    # compute bias
    bias = np.mean((y - E_L)**2)

    # compute variance
    var = np.mean(np.mean((y_tilde_matrix - E_L)**2, axis=1, keepdims=True))

    # compute bootstrap mean of estimators
    estimator_mean = np.sum(bootstrap_estimator, axis=1)/B

    # do some printing for test purposes
    print("VAR: %f" % var)
    print("BIAS: %f" % bias)
    print("Bootstrap mean of MSE: %.4f" % estimator_mean[0])
    print("Bootstrap mean of r2Score: %.4f" % estimator_mean[1])

    return_values = np.zeros((4,1))
    return_values[0] = bias
    return_values[1] = var
    return_values[2] = estimator_mean[0]
    return_values[3] = estimator_mean[1]

    return return_values
