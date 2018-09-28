from utilities import SumOneToN
import numpy as np 

class RidgeLinearModel:
    covariance_matrix = None # covarriance matrix of the model coefficients
    beta = None # coefficients of the modelfunction


    def __init__(this, lmb, k):
        this.lmb = lmb # set lambda of model
        this.k = k # set order of polynomial


    # This function fits the model to the the sample data 
    # using Ridge regression
    #
    # @x: array containing predictors
    # @y: array containing responses
    # @k: the degree of the polynomial to be fitted to the sample data
    # @lmb: lambda, determines the emphasize on minimizing the variance 
    #       of the model
    # 
    def fit(this, x, y):
        # calculate the dimensions of the design matrix
        m = x.shape[0]
        n = SumOneToN(this.k + 1)

        # allocate design matrix
        this.X = np.ones((m, n))

        # compute values of design matrix
        for i in range(m):
            for p in range(this.k):
                for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                    this.X[i][SumOneToN(p + 1) + j] *= x[i][0]**(p 
                            + 1 - j)*x[i][1]**j

        # compute linear regression coefficients
        this.beta = np.linalg.inv(this.X.T.dot(this.X) + 
                this.lmb*np.identity(n)).dot(this.X.T).dot(y)
        
        # compute constant needed later    
        this.qsig = (sum(y-sum(y)/m)/(x.shape[0] - x.shape[1] -1))


    # Predicts and returns the responses of the predictors with 
    # the fitted model if the model is fitted
    # 
    # @x: array containing predictors
    #
    def predict(this, x):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            # allocate meshgrid filled with constant term
            y = np.ones(x.shape[0])*this.beta[0]

            # compute function values
            for p in range(this.k):
                for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                    y += this.beta[SumOneToN(p + 1) 
                            + j]*x[:,0]**(p+1-j)*x[:,1]**j

            return y


    # Computes and returns the covariance matrix of the coefficients of
    # the model function (beta)
    def get_covariance_matrix(this):
        if this.beta is None:
            if this.covariance_matrix is None:
                this.covariance_matrix = np.linalg.inv(this.X.T.dot(this.X))*this.qsig
        else:
    return this.covariance_matrix

    
    def get_var():
        np.ones()
