from utilities import SumOneToN, RSS, MSE, R2Score
import numpy as np 

class RidgeLinearModel:
    rss = None # residual squared sum of model
    rss_updated = False
    mse = None # Mean squared error of model
    mse_updated = False
    r2score = None #R2 score of model
    r2score_updated = False
    covariance_matrix = None # covariance matrix of the model coefficients
    covariance_matrix_updated = False
    beta = None # coefficients of the modelfunction
    x = None # predictors of sampledata
    y = None # responses of sampledata
    y_tilde = None # model predictions for x
    y_tilde_updated = False


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
        # store x ands y for later computations
        this.x = x
        this.y = y

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

        # stored statistical parameters are no longer valid
        this.set_updated_to_false()


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
            y = np.ones((x.shape[0],1))*this.beta[0]

            # compute function values
            for p in range(this.k):
                for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                    y += this.beta[SumOneToN(p + 1) 
                            + j]*x[:,0:1]**(p+1-j)*x[:,1:2]**j

            return y


    # Computes and returns the covariance matrix of the coefficients of
    # the model function (beta)
    def get_covariance_matrix(this):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.covariance_matrix_updated:
                m = this.x.shape[0]
                qsigma = (sum(this.y-sum(this.y)/m)/(this.x.shape[0] - this.x.shape[1] -1))
                this.covariance_matrix = np.linalg.inv(this.X.T.dot(this.X))*qsigma
                covariance_matrix_updated = True

            return this.covariance_matrix

    
    # Computes and returns the variance of the coefficients of the model
    def get_variance_of_coefficients(this):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            var_matrix = this.get_covariance_matrix()
            n = SumOneToN(this.k + 1)
            var_vector = np.ones(n)
            
            for i in range(n):
                var_vector[i] = var_matrix[i][i]

            return var_vector
            
    
    # Returns the residuals of the model squared and summed
    def get_RSS(this):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.rss_updated:
                
                if not this.y_tilde_updated:
                    this.y_tilde = this.predict(this.x)

                this.rss = RSS(this.y, this.y_tilde)
            
            return this.rss
                

    # Returns the mean squared error of the model    
    def get_MSE(this):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.mse_updated:
                
                if not this.y_tilde_updated:
                    this.y_tilde = this.predict(this.x)

                this.mse = MSE(this.y, this.y_tilde)
            
            return this.mse
        

    # Returns the R2 score of the model
    def get_R2Score(this):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.r2score_updated:
                
                if not this.y_tilde_updated:
                    this.y_tilde = this.predict(this.x)

                this.r2score = R2Score(this.y, this.y_tilde)
            
            return this.r2score


    def set_updated_to_false(this):
        rss_updated = False
        mse_updated = False
        r2score_updated = False
        covariance_matrix_updated = False
        y_tilde_updated = False


