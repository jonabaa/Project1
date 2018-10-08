# DELETE
# k-th order polynomial DELETE
def Polynome(x, y, k, beta):
    if beta.size != SumOneToN(k + 1):
        print("Function Polynom says: Bad input data.")
        return 0
    else:
        # allocate meshgrid of appropriate shape filles with constant term
        z = np.ones(x.shape)*beta[0]

        # compute function values
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                z += beta[SumOneToN(p + 1) + j]*x**(p+1-j)*y**j

        return z

# DELETE
# Fits a k-th order polynomial, p:R^2 -> R, to the given data x, y
# using Ridge regression with lambda=lmb
def RidgeReg(x, y, k, lmb):
    print(x.shape)
    # calculate the dimensions of the design matrix
    m = x.shape[0]
    n = SumOneToN(k + 1)

    # allocate design matrix
    X = np.ones((m, n))

    # compute values of design matrix
    for i in range(m):
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                X[i][SumOneToN(p + 1) + j] *= x[i][0]**(p+1-j)*x[i][1]**j

    # compute linear regression coefficients
    beta = np.linalg.inv(X.T.dot(X) + lmb*np.identity(n)).dot(X.T).dot(y)

    covar_matrix = np.linalg.inv(X.T.dot(X)) #*(sum(y-sum(y)/m)/(x.shape[0] - x.shape[1] -1))

    return beta

# DELETE
# Fits a k-th order polynomial, p:R^2 -> R, to the given data x, y
# using Lasso regression with lambda=lmb
def LassoReg(x, y, k, lmb):
    # calculate the dimensions of the design matrix
    m = x.shape[0]
    n = SumOneToN(k + 1)

    # allocate design matrix
    X = np.ones((m, n))

    # compute values of design matrix
    for i in range(m):
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                X[i][SumOneToN(p + 1) + j] *= x[i][0]**(p+1-j)*x[i][1]**j

    #compute lasso-coefficients using scikitlearn
    lasso = linear_model.Lasso(alpha=lmb)
    lasso.fit(X, y)
    beta = lasso.coef_

    return beta

# Bootstrap with B resamples
#
# s = sample data
# f = Regressionfunction ((RidgeRwg or LassoReg)
# k = order of polynomial for regression function to fit
# lmb = lambda (set to 0 for OLS-regression)
# p = list of parameter estimator functions
# B = number of bootstrap-samples
#
def Bootstrap(s, f, k, lmb, p, B):
    # allocate arrays for storing bootstrap estimators
    bootstrap_estimator = np.zeros((len(p), B))

    for b in range(B):
        # draw a random bootstrap sample with replacement
        bs_s = s[np.random.choice(s.shape[0], s.shape[0]),:]

        # compute model
        x = bs_s[:,0:2]
        y = bs_s[:,2]
        beta = f(x, y, k, lmb)

        # compute y_tilde
        y_tilde = Polynome(x[:,0], x[:,1], k, beta)

        # compute estimators of parameters
        for i in range(len(p)):
            bootstrap_estimator[i,b] = p[i](y, y_tilde)

    # compute bootstrap mean of estimators
    estimator_mean = np.sum(bootstrap_estimator, axis=1)/B

    return estimator_mean


# Bootstrap with B resamples
#
# s = sample data
# f = Regressionfunction ((RidgeRwg or LassoReg)
# k = order of polynomial for regression function to fit
# lmb = lambda (set to 0 for OLS-regression)
# B = number of bootstrap-samples
#
def Bootstrap2(s, f, k, lmb, B):
    # allocate arrays for storing bootstrap estimators
    bootstrap_estimator = np.zeros((2, B))

    for b in range(B):
        # draw a random bootstrap sample with replacement
        bs_s = s[np.random.choice(s.shape[0], s.shape[0]),:]

        # compute model
        x = bs_s[:,0:2]
        y = bs_s[:,2]
        beta = f(x, y, k, lmb)

        # compute y_tilde
        y_tilde = Polynome(x[:,0:1], x[:,1:2], k, beta)
        if b == 0:
            y_tilde_matrix = y_tilde
        else:
            y_tilde_matrix = np.concatenate([y_tilde_matrix, y_tilde], axis=1)
        # compute estimators of parameters
        bootstrap_estimator[0,b] = MSE(y, y_tilde_matrix[:,b])
        bootstrap_estimator[1,b] = R2Score(y, y_tilde_matrix[:,b])
    """
    # compute variance
    for b in range(B):
        y_tilde_matrix[:,b] = (y_tilde_matrix[:,b] - np.sum(y_tilde_matrix, axis=1)/B)**2
    var = np.sum(y_tilde_matrix, axis=1)/B
    var = sum(var)
    print(var)
    """
    #compute expected value in x over the bootstrapsamples
    E_L = (np.sum(y_tilde_matrix, axis=1)/B).reshape((len(y_tilde),1))

    # compute variance
    var = sum(np.sum((y_tilde_matrix - E_L)**2, axis=1)/B)

    # compute bias
    bias = np.sum((y - E_L)**2)

    # compute bootstrap mean of estimators
    estimator_mean = np.sum(bootstrap_estimator, axis=1)/B

    # do some printing for test purposes
    print("VAR: %.4f." % var)
    print("BIAS: %.4f." % bias)
    print("Bootstrap mean of MSE: %.4f." % estimator_mean[0])
    print("Bootstrap mean of r2Score: %.4f." % estimator_mean[1])

    return_values = np.zeros((4,1))
    return_values[0] = bias
    return_values[1] = var
    return_values[2] = estimator_mean[0]
    return_values[3] = estimator_mean[1]

    return return_values

def y_predict(x, k, beta):
    # Given x, k and beta, returns predicted y
    y_tilde = np.ones((x.shape[0], 1))*beta[0]

    for i in range(x.shape[0]):
        for p in range(k):
            for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                y_tilde[i] += beta[SumOneToN(p + 1) + j]*x[i][0]**(p+1-j)*x[i][1]**j

    return y_tilde
